#!/usr/bin/env python3
"""
finetune_chemberta_baseline.py

Fine-tunes DeepChem/ChemBERTa-77M-MTR (Ahmad et al. 2022, ChemBERTa-2 --
the MTR/multi-task-regression pretrained variant, which their own paper
found beats the MLM variant on downstream tasks; MTR pretraining predicts
~200 real molecular descriptors, philosophically the same "pretrain on
computable descriptors" idea as CHEMELEON, just via a transformer over
SMILES tokens instead of a D-MPNN over the molecular graph) as a binary
classifier on CYP450 substrate data -- the real head-to-head baseline
behind model/registry.json's cyp2c9-substrate-v1/cyp2e1-substrate-v1
metrics.note "CHEMBERTA COMPARISON" section (verdict: CHEMELEON won on
every metric except a near-tied F1/MCC, and is also the cheaper browser-
deployment path since it reuses this project's existing D-MPNN inference
engine -- see that note for the full numbers and reasoning).

Uses the EXACT same SCAFFOLD_BALANCED train/val/test splits chemprop
itself produced (verified byte-identical for the same --data-seed --
chemprop's splitter is deterministic), for a genuinely apples-to-apples
comparison against every chemprop configuration already tried. Produce
those split files first with, e.g.:
  chemprop train --data-path data/cyp_substrate/cyp_substrate_2c9.csv \
    --smiles-columns smiles --target-columns label --task-type classification \
    --split SCAFFOLD_BALANCED --split-sizes 0.8 0.1 0.1 \
    --data-seed 0 --pytorch-seed 0 --epochs 2 --save-smiles-splits \
    --output-dir /tmp/splitextract_2c9_seed0
(any hyperparameters/epoch count work here -- --save-smiles-splits's
output only depends on --split/--split-sizes/--data-seed, not on how
long training actually runs).

Class imbalance handled via WeightedRandomSampler (inverse class
frequency), the standard PyTorch-native equivalent of chemprop's
--class-balance (which does the same "see positives and negatives
equally often per batch" thing via a different sampling mechanism).

Usage:
  finetune_chemberta_baseline.py <train_smiles_csv> <val_smiles_csv> <test_smiles_csv> \
      <out_prefix> --labels-csv <original_labeled_csv> [--seed N] [--epochs N]
"""
import argparse
import csv
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef

MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"


def load_split(split_csv_path, labels_by_smiles):
    """split_csv_path is chemprop --save-smiles-splits output (smiles column
    only); join back against the original labeled CSV for true labels."""
    with open(split_csv_path) as f:
        smiles = [r["smiles"] for r in csv.DictReader(f)]
    return smiles, [labels_by_smiles[s] for s in smiles]


def load_labels_csv(path):
    with open(path) as f:
        return {r["smiles"]: int(float(r["label"])) for r in csv.DictReader(f)}


class SmilesDataset(Dataset):
    def __init__(self, smiles, labels, tokenizer, max_len=128):
        self.enc = tokenizer(smiles, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.enc["input_ids"][idx],
            "attention_mask": self.enc["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("train_csv")
    p.add_argument("val_csv")
    p.add_argument("test_csv")
    p.add_argument("out_prefix")
    p.add_argument("--labels-csv", required=True, help="original labeled CSV (smiles,label) to join split files against")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

    labels_by_smiles = load_labels_csv(args.labels_csv)
    train_smiles, train_labels = load_split(args.train_csv, labels_by_smiles)
    val_smiles, val_labels = load_split(args.val_csv, labels_by_smiles)
    test_smiles, test_labels = load_split(args.test_csv, labels_by_smiles)
    print(f"train={len(train_smiles)} val={len(val_smiles)} test={len(test_smiles)} "
          f"train_pos={sum(train_labels)}", file=sys.stderr)

    train_ds = SmilesDataset(train_smiles, train_labels, tokenizer)
    val_ds = SmilesDataset(val_smiles, val_labels, tokenizer)
    test_ds = SmilesDataset(test_smiles, test_labels, tokenizer)

    # WeightedRandomSampler: inverse class frequency, same "see both
    # classes equally often" effect as chemprop's --class-balance.
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)

    def evaluate(loader):
        model.eval()
        all_probs, all_labels, total_loss, n = [], [], 0.0, 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                total_loss += out.loss.item() * batch["labels"].size(0)
                n += batch["labels"].size(0)
                probs = torch.softmax(out.logits, dim=-1)[:, 1]
                all_probs.extend(probs.cpu().numpy().tolist())
                all_labels.extend(batch["labels"].cpu().numpy().tolist())
        return total_loss / n, all_labels, all_probs

    best_val_loss = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        val_loss, val_true, val_probs = evaluate(val_loader)
        val_roc = roc_auc_score(val_true, val_probs) if len(set(val_true)) > 1 else float("nan")
        print(f"epoch {epoch}: val_loss={val_loss:.4f} val_roc={val_roc:.4f}", file=sys.stderr)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_loss, test_true, test_probs = evaluate(test_loader)
    roc = roc_auc_score(test_true, test_probs)
    prc = average_precision_score(test_true, test_probs)
    test_bin = [1 if p >= 0.5 else 0 for p in test_probs]
    f1 = f1_score(test_true, test_bin)
    mcc = matthews_corrcoef(test_true, test_bin)
    print(f"RESULT seed={args.seed} best_val_loss={best_val_loss:.4f} "
          f"test_ROC={roc:.4f} test_PRC={prc:.4f} test_F1={f1:.4f} test_MCC={mcc:.4f}")

    with open(f"{args.out_prefix}_test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles", "true_label", "pred_prob"])
        for smi, t, p in zip(test_smiles, test_true, test_probs):
            w.writerow([smi, t, p])


if __name__ == "__main__":
    main()
