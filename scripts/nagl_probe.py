"""
Run this in your nagl-mbis conda/mamba environment. Doesn't need network --
just inspects a checkpoint you already have locally.
"""
import torch

path = "naglmbis/data/models/nagl-v1-mbis.ckpt"  # adjust to your actual path
data = torch.load(path, map_location="cpu", weights_only=False)

print("=== top-level keys ===")
print(list(data.keys()))

hp = data["hyper_parameters"]
print("\n=== hyper_parameters ===")
for k, v in hp.items():
    print(f"{k}: {v}")

print("\n=== state_dict keys/shapes ===")
for k, v in data["state_dict"].items():
    print(k, tuple(v.shape) if hasattr(v, "shape") else v)

# Try an actual forward pass and inspect what compute_properties expects/returns
try:
    from naglmbis.models import load_charge_model
    model = load_charge_model(charge_model="nagl-v1-mbis")
    from rdkit import Chem
    mol = Chem.MolFromSmiles("CCO")
    mol = Chem.AddHs(mol)
    out = model.compute_properties(mol)
    print("\n=== compute_properties output ===")
    print(type(out))
    print(out)
except Exception as e:
    print("\ncompute_properties test failed:", type(e).__name__, e)
