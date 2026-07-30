"""
Run in your working nagl-mbis environment. Generates real reference MBIS
charge predictions on a handful of simple test molecules, for diffing
against the JS port.
"""
from naglmbis.models import load_charge_model
from rdkit import Chem
import json

model = load_charge_model(charge_model="nagl-v1-mbis")

test_smiles = [
    "CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "c1ccc2ccccc2c1",
    "CC(=O)[O-]", "CCN", "CC(C)=O", "c1ccncc1",
]

results = {}
for smi in test_smiles:
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    charges = model.compute_properties(mol)["mbis-charges"]
    charges_list = charges.detach().flatten().tolist() if hasattr(charges, "detach") else list(charges)
    results[smi] = charges_list
    print(smi, charges_list)

with open("nagl_reference_values.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nwrote nagl_reference_values.json")
