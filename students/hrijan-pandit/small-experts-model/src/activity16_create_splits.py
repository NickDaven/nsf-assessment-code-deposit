"""
Activity 16 — Create Calibration and Held-Out Evaluation Splits
================================================================
Run AFTER activity15_organize_evaluation.py
"""

import pandas as pd
import hashlib, json, os
from sklearn.model_selection import train_test_split

PROCESSED = "/workspace/eir-project/data/evaluation/processed"
OUT_DIR   = "/workspace/eir-project/data/evaluation/splits"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(f"{PROCESSED}/aligned_evaluation.csv")

# 50/50 calibration vs held-out split, stratified by label
calib, held_out = train_test_split(
    df, test_size=0.5, stratify=df["label"], random_state=42)

calib.to_csv(f"{OUT_DIR}/calibration_split.csv", index=False)
held_out.to_csv(f"{OUT_DIR}/held_out_split.csv", index=False)

# Checksums
def sha256(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    "calibration": {"rows": len(calib),   "sha256": sha256(f"{OUT_DIR}/calibration_split.csv")},
    "held_out":    {"rows": len(held_out), "sha256": sha256(f"{OUT_DIR}/held_out_split.csv")},
    "split_rule":  "50/50 stratified by label, seed=42"
}
with open(f"{OUT_DIR}/split_manifest.json","w") as f:
    json.dump(manifest, f, indent=2)

print(f"Calibration: {len(calib)} rows")
print(f"Held-out:    {len(held_out)} rows")
print("Split manifest saved. Activity 16 complete.")
