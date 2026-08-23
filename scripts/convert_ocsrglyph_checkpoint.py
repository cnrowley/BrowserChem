#!/usr/bin/env python3
"""
convert_ocsrglyph_checkpoint.py

Converts EdisonScientific/OCSRGlyph's PyTorch checkpoint (model.pth, a
Swin-B image encoder + 6-layer Transformer SMILES decoder, ~357MB fp32)
into two ONNX graphs (encoder.onnx, decoder.onnx) plus a manifest.json +
vocab_chars.json pair js/ocsrglyph-model.js loads directly in the browser
via ONNX Runtime Web -- no server, no PyTorch at runtime.

Unlike this project's other model-conversion scripts (chemprop, NAGL),
which hand-extract raw weight tensors into a project-authored JS forward
pass, this one does NOT reimplement Swin's windowed/shifted-window
attention from scratch in JS -- that architecture is far more involved
than the message-passing nets this project's other GNN engines
reimplement, and ONNX Runtime Web (already loaded in this app for a
legacy manual-load demo path, js/gnn-inference.js) is the right tool for
a fixed-shape vision-transformer + autoregressive-decoder pair like this
one. The two exported graphs mirror the reference `_greedy_batch`
decoding loop exactly (glyph/ocsr/predict.py in the upstream repo): the
encoder runs once per image, the decoder is called repeatedly with the
growing token sequence so far (no KV-cache in the reference
implementation, so none is added here -- a direct, low-risk port).

Architecture/preprocessing/decoding details below are confirmed directly
from the real upstream source (github.com/EdisonScientific/glyph,
Apache-2.0) and the checkpoint's own recorded `recipe` dict, not
inferred: glyph/ocsr/model.py (OCSREncoder/OCSRDecoder/OCSRModel),
glyph/ocsr/predict.py (_preprocess_image, _greedy_batch), glyph/ocsr/
postprocess.py (postprocess_smiles), glyph/ocsr/smiles_tokenizer.py
(CharSmilesTokenizer -- a trivial flat char->id dict, vendored here as
vocab_chars.json, not reimplemented).

Usage:
    pip install torch timm huggingface_hub onnx onnxruntime onnxconverter-common
    python3 convert_ocsrglyph_checkpoint.py [--checkpoint model.pth] output_dir/

If --checkpoint is omitted, downloads EdisonScientific/OCSRGlyph's
model.pth from the Hugging Face Hub (357MB). The exported ONNX graphs are
converted to fp16 (halving the ~180MB-vs-357MB browser download) and
validated against the original fp32 PyTorch model's own output on a real
test image before being trusted -- see --test-image.
"""

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

VOCAB_URL = "https://raw.githubusercontent.com/EdisonScientific/glyph/main/glyph/ocsr/vocab/vocab_chars.json"
DEFAULT_TEST_IMAGE_URL = "https://raw.githubusercontent.com/EdisonScientific/glyph/main/examples/imatinib.png"
# Ground truth from the upstream repo's own examples/README.md ("Every
# command in that section is verified against these files").
DEFAULT_TEST_IMAGE_EXPECTED_SMILES = (
    "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
)


def build_model(cfg):
    """Re-creates OCSREncoder/OCSRDecoder/OCSRModel exactly as defined in
    the real glyph/ocsr/model.py (Apache-2.0, github.com/EdisonScientific/
    glyph) -- inlined here rather than depending on the full `glyph`
    package (which pulls in training-only deps like Indigo/wandb this
    conversion has no use for)."""

    import timm
    import torch
    from torch import nn

    class OCSREncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = timm.create_model(
                cfg["encoder_name"],
                pretrained=False,
                features_only=True,
                img_size=cfg["input_size"],
            )
            feat_dim = self.backbone.feature_info.channels()[-1]
            self.proj = nn.Linear(feat_dim, cfg["embed_dim"])

        def forward(self, image):
            feats = self.backbone(image)[-1]
            target_c = self.backbone.feature_info.channels()[-1]
            if feats.dim() == 4 and feats.shape[-1] == target_c and feats.shape[1] != target_c:
                feats = feats.permute(0, 3, 1, 2).contiguous()
            _b, _c, _h, _w = feats.shape
            seq = feats.flatten(2).transpose(1, 2)
            return self.proj(seq)

    class OCSRDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_embed = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"], padding_idx=cfg["pad_id"])
            self.pos_embed = nn.Embedding(cfg["max_len"], cfg["embed_dim"])
            layer = nn.TransformerDecoderLayer(
                d_model=cfg["embed_dim"],
                nhead=cfg["dec_attn_heads"],
                dim_feedforward=cfg["dec_ff_dim"],
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(layer, num_layers=cfg["dec_num_layers"])
            self.head = nn.Linear(cfg["embed_dim"], cfg["vocab_size"])

        def forward(self, tokens, memory):
            b, t = tokens.shape
            pos = torch.arange(t, device=tokens.device).unsqueeze(0).expand(b, t)
            x = self.tok_embed(tokens) + self.pos_embed(pos)
            causal = torch.triu(torch.full((t, t), float("-inf"), device=tokens.device), diagonal=1)
            # tgt_is_causal=True is a pure optimization hint (the mask we
            # pass is unchanged/still applied) -- without it,
            # nn.TransformerDecoder's internal _detect_is_causal_mask does
            # a DATA-DEPENDENT tensor comparison to guess causality, which
            # torch.export's dynamo-based ONNX exporter can't trace
            # through when seq_len is a dynamic symbolic dimension
            # (confirmed directly: export failed with
            # GuardOnDataDependentSymNode without this). Numerically a
            # no-op -- validated against the unmodified upstream forward's
            # own output on a real image before this change was made.
            out = self.decoder(tgt=x, memory=memory, tgt_mask=causal, tgt_is_causal=True)
            return self.head(out)

    class OCSRModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = OCSREncoder()
            self.decoder = OCSRDecoder()

        def forward(self, image, tokens):
            memory = self.encoder(image)
            return self.decoder(tokens, memory)

    return OCSRModel()


def load_vocab(vocab_path):
    if vocab_path and Path(vocab_path).exists():
        return json.loads(Path(vocab_path).read_text())
    print(f"Fetching vocab_chars.json from {VOCAB_URL} ...")
    with urllib.request.urlopen(VOCAB_URL) as resp:
        return json.loads(resp.read().decode("utf-8"))


def preprocess_image(pil_image, input_size):
    """Matches predict.py's _preprocess_image exactly: RGB, resize
    bilinear to (input_size, input_size), [-1,1] normalize, CHW."""
    import numpy as np

    img = pil_image.convert("RGB")
    if img.size != (input_size, input_size):
        from PIL import Image

        img = img.resize((input_size, input_size), Image.BILINEAR)
    arr = np.asarray(img, dtype="float32") / 255.0
    arr = (arr - 0.5) / 0.5
    return arr.transpose(2, 0, 1)[None, ...]  # [1,3,H,W]


def greedy_decode_onnx(encoder_sess, decoder_sess, image_np, cfg):
    """Reference-matching greedy decode (predict.py's _greedy_batch),
    run through ONNX Runtime instead of PyTorch -- used both to validate
    the export and as a template for js/ocsrglyph-model.js's own loop."""
    import numpy as np

    memory = encoder_sess.run(None, {"image": image_np.astype("float32")})[0]
    ids = [cfg["sos_id"]]
    for _ in range(cfg["max_len"] - 1):
        tokens = np.array([ids], dtype="int64")
        logits = decoder_sess.run(None, {"tokens": tokens, "memory": memory.astype("float32")})[0]
        nxt = int(np.argmax(logits[0, -1, :]))
        ids.append(nxt)
        if nxt == cfg["eos_id"]:
            break
    return ids


def decode_ids_to_smiles(ids, itos, cfg):
    chars = []
    for i in ids:
        if i == cfg["eos_id"]:
            break
        if i in (cfg["pad_id"], cfg["sos_id"]):
            continue
        chars.append(itos.get(i, "?"))
    return "".join(chars)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", help="directory to write encoder.onnx/decoder.onnx/manifest.json/vocab_chars.json into")
    parser.add_argument("--checkpoint", default=None, help="path to a local model.pth; downloads from HF if omitted")
    parser.add_argument("--vocab", default=None, help="path to a local vocab_chars.json; fetches from GitHub if omitted")
    parser.add_argument("--test-image", default=None, help="local image path to validate against; downloads examples/imatinib.png if omitted")
    parser.add_argument("--skip-fp16", action="store_true", help="keep fp32 ONNX output (skip the fp16 conversion pass)")
    parser.add_argument("--max-len", type=int, default=256, help="greedy-decode cap (matches predict.py's MAX_DECODE_LEN)")
    args = parser.parse_args()

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from onnxconverter_common import float16 as onnx_float16
    from PIL import Image

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        from huggingface_hub import hf_hub_download

        print("Downloading EdisonScientific/OCSRGlyph model.pth from Hugging Face Hub (357MB)...")
        checkpoint_path = hf_hub_download("EdisonScientific/OCSRGlyph", "model.pth")

    print(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "model" not in state:
        sys.exit("Checkpoint must be a dict containing a 'model' state_dict.")
    recipe = state.get("recipe") or {}
    if state.get("quantized"):
        sys.exit(f"This checkpoint is pre-quantized ({state['quantized']!r}); pass the fp32 model.pth instead.")

    cfg = {
        "encoder_name": recipe.get("encoder_name", "swin_base_patch4_window12_384"),
        "input_size": recipe.get("input_size", 384),
        "embed_dim": recipe.get("embed_dim", 256),
        "dec_num_layers": recipe.get("dec_num_layers", 6),
        "dec_attn_heads": recipe.get("dec_attn_heads", 8),
        "dec_ff_dim": recipe.get("dec_ff_dim", 1024),
        "vocab_size": recipe.get("vocab_size", 101),
        "pad_id": recipe.get("pad_id", 0),
        "sos_id": recipe.get("sos_id", 1),
        "eos_id": recipe.get("eos_id", 2),
        "max_len": min(recipe.get("max_target_len", 480), args.max_len),
    }
    print("Resolved config:", json.dumps(cfg, indent=2))

    model = build_model(cfg)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if missing or unexpected:
        sys.exit(
            "State dict did not load cleanly against the rebuilt architecture "
            f"-- missing={missing}, unexpected={unexpected}. The upstream model.py "
            "may have changed; re-check it against this script's build_model()."
        )
    model.eval()

    vocab = load_vocab(args.vocab)
    itos = {v: k for k, v in vocab.items()}
    for tok, expected in [("<pad>", 0), ("<sos>", 1), ("<eos>", 2)]:
        if vocab.get(tok) != expected:
            sys.exit(f"Unexpected vocab: {tok!r} should be id {expected}, got {vocab.get(tok)}")

    # ---- Export encoder: image [1,3,H,W] -> memory [1,HW,embed_dim] ----
    dummy_image = torch.zeros(1, 3, cfg["input_size"], cfg["input_size"], dtype=torch.float32)
    encoder_path = output_dir / "encoder.onnx"
    print(f"Exporting encoder -> {encoder_path}")
    torch.onnx.export(
        model.encoder,
        (dummy_image,),
        str(encoder_path),
        input_names=["image"],
        output_names=["memory"],
        opset_version=17,
        dynamo=False,
    )

    # ---- Export decoder: tokens [1,T] + memory [1,HW,embed_dim] -> logits [1,T,vocab] ----
    # The legacy TorchScript-based exporter (dynamo=False) traces
    # nn.MultiheadAttention's internal head-split reshape with the dummy
    # input's CONCRETE seq_len baked in, regardless of dynamic_axes --
    # confirmed directly: it produced a decoder.onnx that only worked for
    # tokens of length exactly len(dummy_tokens), failing ONNX Runtime's
    # own Reshape node at any other length. The newer torch.export-based
    # exporter (dynamo=True) propagates a real symbolic seq_len through
    # that reshape instead, which is what a token-by-token greedy decode
    # loop (growing seq_len every step) actually needs.
    with torch.no_grad():
        dummy_memory = model.encoder(dummy_image)
    dummy_tokens = torch.zeros(1, 3, dtype=torch.long)
    seq_len = torch.export.Dim("seq_len", min=1, max=cfg["max_len"])
    decoder_path = output_dir / "decoder.onnx"
    print(f"Exporting decoder -> {decoder_path}")
    torch.onnx.export(
        model.decoder,
        (dummy_tokens, dummy_memory),
        str(decoder_path),
        input_names=["tokens", "memory"],
        output_names=["logits"],
        dynamic_shapes={"tokens": {1: seq_len}, "memory": None},
        opset_version=17,
        dynamo=True,
    )

    onnx.checker.check_model(str(encoder_path))
    onnx.checker.check_model(str(decoder_path))
    print("ONNX structural checks passed.")

    # ---- Validate fp32 ONNX vs PyTorch (and vs known ground truth) on a real image ----
    if args.test_image:
        test_img = Image.open(args.test_image)
    else:
        print(f"Fetching test image from {DEFAULT_TEST_IMAGE_URL} ...")
        tmp_img_path = output_dir / "_test_image.png"
        urllib.request.urlretrieve(DEFAULT_TEST_IMAGE_URL, tmp_img_path)
        test_img = Image.open(tmp_img_path)

    image_np = preprocess_image(test_img, cfg["input_size"])

    with torch.no_grad():
        torch_ids = [cfg["sos_id"]]
        image_t = torch.from_numpy(image_np)
        memory_t = model.encoder(image_t)
        for _ in range(cfg["max_len"] - 1):
            tokens_t = torch.tensor([torch_ids], dtype=torch.long)
            logits_t = model.decoder(tokens_t, memory_t)
            nxt = int(logits_t[0, -1, :].argmax())
            torch_ids.append(nxt)
            if nxt == cfg["eos_id"]:
                break
    torch_smiles = decode_ids_to_smiles(torch_ids, itos, cfg)
    print(f"PyTorch (fp32) decoded SMILES: {torch_smiles}")

    encoder_sess_fp32 = ort.InferenceSession(str(encoder_path))
    decoder_sess_fp32 = ort.InferenceSession(str(decoder_path))
    onnx_ids = greedy_decode_onnx(encoder_sess_fp32, decoder_sess_fp32, image_np, cfg)
    onnx_smiles = decode_ids_to_smiles(onnx_ids, itos, cfg)
    print(f"ONNX (fp32) decoded SMILES:    {onnx_smiles}")

    if torch_smiles != onnx_smiles:
        sys.exit("PyTorch and fp32-ONNX decoded SMILES DIFFER -- export is not faithful, stopping before fp16/manifest.")
    print("PyTorch vs fp32-ONNX: MATCH.")

    if not args.test_image and torch_smiles.replace(" ", "") != DEFAULT_TEST_IMAGE_EXPECTED_SMILES:
        # Try RDKit canonical-form comparison before treating this as fatal --
        # the raw decoded string need not byte-match the README's canonical
        # form, but a real recognition failure should still stop the script.
        try:
            from rdkit import Chem

            m1 = Chem.MolFromSmiles(torch_smiles)
            m2 = Chem.MolFromSmiles(DEFAULT_TEST_IMAGE_EXPECTED_SMILES)
            same = m1 is not None and m2 is not None and Chem.MolToSmiles(m1) == Chem.MolToSmiles(m2)
        except ImportError:
            same = False
        if not same:
            sys.exit(
                f"Decoded SMILES does not match the known-correct imatinib.png reference.\n"
                f"  got:      {torch_smiles}\n"
                f"  expected: {DEFAULT_TEST_IMAGE_EXPECTED_SMILES}\n"
                "Stopping -- something is wrong with the rebuilt architecture or checkpoint loading."
            )
        print("Canonical-SMILES match against the known imatinib.png reference: MATCH.")

    # ---- fp16 conversion pass (post-hoc, on the ONNX graph -- not a
    # torch.onnx.export(..., half()) path, which is more prone to
    # unsupported-op issues for a Swin backbone) ----
    if not args.skip_fp16:
        for name, path in [("encoder", encoder_path), ("decoder", decoder_path)]:
            print(f"Converting {name}.onnx to fp16...")
            m = onnx.load(str(path))
            m16 = onnx_float16.convert_float_to_float16(m, keep_io_types=True)
            onnx.save(m16, str(path))

        encoder_sess_fp16 = ort.InferenceSession(str(encoder_path))
        decoder_sess_fp16 = ort.InferenceSession(str(decoder_path))
        onnx16_ids = greedy_decode_onnx(encoder_sess_fp16, decoder_sess_fp16, image_np, cfg)
        onnx16_smiles = decode_ids_to_smiles(onnx16_ids, itos, cfg)
        print(f"ONNX (fp16) decoded SMILES:    {onnx16_smiles}")
        if onnx16_smiles != torch_smiles:
            try:
                from rdkit import Chem

                m1 = Chem.MolFromSmiles(onnx16_smiles)
                m2 = Chem.MolFromSmiles(torch_smiles)
                same = m1 is not None and m2 is not None and Chem.MolToSmiles(m1) == Chem.MolToSmiles(m2)
            except ImportError:
                same = False
            if not same:
                sys.exit(
                    "fp16 ONNX decoded SMILES differs from the validated fp32 result on the "
                    "test image -- fp16 conversion introduced a real accuracy regression. "
                    "Re-run with --skip-fp16 and ship fp32, or investigate before shipping fp16."
                )
            print("fp16 result differs byte-for-byte but is chemically identical (canonical match): OK.")
        else:
            print("PyTorch vs fp16-ONNX: MATCH.")

    # ---- Write vocab + manifest ----
    (output_dir / "vocab_chars.json").write_text(json.dumps(vocab, indent=2))
    manifest = {
        "modelId": "ocsrglyph",
        "source": "EdisonScientific/OCSRGlyph",
        "sourceUrl": "https://huggingface.co/EdisonScientific/OCSRGlyph",
        "codeUrl": "https://github.com/EdisonScientific/glyph",
        "license": "Apache-2.0",
        "inputSize": cfg["input_size"],
        "embedDim": cfg["embed_dim"],
        "vocabSize": cfg["vocab_size"],
        "padId": cfg["pad_id"],
        "sosId": cfg["sos_id"],
        "eosId": cfg["eos_id"],
        "maxLen": cfg["max_len"],
        "precision": "fp32" if args.skip_fp16 else "fp16",
        "validated": {
            "testImage": "examples/imatinib.png (upstream repo)" if not args.test_image else args.test_image,
            "decodedSmiles": torch_smiles,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    tmp_img = output_dir / "_test_image.png"
    if tmp_img.exists() and not args.test_image:
        tmp_img.unlink()

    enc_size = encoder_path.stat().st_size
    dec_size = decoder_path.stat().st_size
    print(f"\nDone. encoder.onnx={enc_size/1e6:.1f}MB decoder.onnx={dec_size/1e6:.1f}MB total={(enc_size+dec_size)/1e6:.1f}MB")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
