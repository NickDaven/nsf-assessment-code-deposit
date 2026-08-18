"""
Activity 20 — Held-Out Human-Machine Evaluation
================================================
REQUIRES: Activities 16, 18, 19 outputs

Run: python3 activity20_held_out_evaluation.py
"""
import json, os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix

SPLITS_DIR     = "/workspace/eir-project/data/evaluation/splits"
RESULTS_DIR    = "/workspace/eir-project/data/results"
THRESHOLDS_DIR = "/workspace/eir-project/data/thresholds"
OUT_DIR        = "/workspace/eir-project/data/results/evaluation"
os.makedirs(OUT_DIR, exist_ok=True)

held_out   = pd.read_csv(f"{SPLITS_DIR}/held_out_split.csv")
with open(f"{RESULTS_DIR}/distilbert_predictions.json") as f:
    predictions = json.load(f)
with open(f"{THRESHOLDS_DIR}/thresholds_distilbert.json") as f:
    thresholds = json.load(f)

pred_lookup = {p["response_id"]: p["rubric_scores"] for p in predictions}
categories  = held_out["category"].unique().tolist()

all_results = []
for cat in categories:
    cat_df  = held_out[held_out["category"] == cat]
    y_true  = (cat_df["label"] == "present").astype(int).tolist()
    thresh  = thresholds.get(cat, {}).get("threshold", 0.5)
    y_pred  = []
    for _, row in cat_df.iterrows():
        scores = pred_lookup.get(row["response_id"], [])
        score  = next((s["best_score"] for s in scores
                      if cat.lower() in s["rubric"].lower()), 0.0)
        y_pred.append(1 if score >= thresh else 0)

    result = {
        "category":  cat,
        "threshold": thresh,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "kappa":     round(cohen_kappa_score(y_true, y_pred), 4) if len(set(y_true)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }
    all_results.append(result)
    print(f"{cat}: P={result['precision']} R={result['recall']} F1={result['f1']}")

macro_f1 = sum(r["f1"] for r in all_results) / len(all_results)
print(f"\nMacro F1: {macro_f1:.4f}")

with open(f"{OUT_DIR}/held_out_results_distilbert.json", "w") as f:
    json.dump({"macro_f1": round(macro_f1,4), "categories": all_results}, f, indent=2)
print("Activity 20 complete.")
