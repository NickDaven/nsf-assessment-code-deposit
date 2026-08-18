"""
Activity 19 — Calibrate Category-Specific Decision Thresholds
=============================================================
REQUIRES: Activity 15 and 16 outputs (real evaluation data)

DROP FILES:
  /workspace/eir-project/data/evaluation/splits/calibration_split.csv
  /workspace/eir-project/data/results/<model>_predictions.json

Then run: python3 activity19_calibrate_thresholds.py
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

SPLITS_DIR  = "/workspace/eir-project/data/evaluation/splits"
RESULTS_DIR = "/workspace/eir-project/data/results"
OUT_DIR     = "/workspace/eir-project/data/thresholds"
os.makedirs(OUT_DIR, exist_ok=True)

# Load calibration split
calib = pd.read_csv(f"{SPLITS_DIR}/calibration_split.csv")
categories = calib["category"].unique().tolist()

# Load model predictions
with open(f"{RESULTS_DIR}/distilbert_predictions.json") as f:
    predictions = json.load(f)

pred_lookup = {p["response_id"]: p["rubric_scores"] for p in predictions}

thresholds = {}
for cat in categories:
    cat_df = calib[calib["category"] == cat]
    y_true = (cat_df["label"] == "present").astype(int).tolist()

    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred = []
        for _, row in cat_df.iterrows():
            rid = row["response_id"]
            scores = pred_lookup.get(rid, [])
            score = next((s["best_score"] for s in scores
                         if cat.lower() in s["rubric"].lower()), 0.0)
            y_pred.append(1 if score >= thresh else 0)
        if len(set(y_true)) > 1:
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, thresh

    thresholds[cat] = {"threshold": round(float(best_thresh), 2), "calibration_f1": round(best_f1, 4)}
    print(f"{cat}: threshold={best_thresh:.2f}, F1={best_f1:.4f}")

with open(f"{OUT_DIR}/thresholds_distilbert.json", "w") as f:
    json.dump(thresholds, f, indent=2)
print("\nThresholds saved. Activity 19 complete.")
