"""
Activity 21 — Error Analysis and Size-Accuracy Tradeoff
========================================================
REQUIRES: Activity 20 outputs for all models

Run: python3 activity21_error_analysis.py
"""
import json, os
import pandas as pd

RESULTS_DIR = "/workspace/eir-project/data/results/evaluation"
OUT_DIR     = "/workspace/eir-project/data/results/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# Model ladder with known stats
model_ladder = [
    {"model": "TinyBERT",   "params_M": 14.4, "memory_MB": 55,    "train_loss": 5.38, "results_file": "held_out_results_tinybert.json"},
    {"model": "DistilBERT", "params_M": 66.4, "memory_MB": 265.5, "train_loss": 2.36, "results_file": "held_out_results_distilbert.json"},
    {"model": "BERT-base",  "params_M": 109.5,"memory_MB": 437.9, "train_loss": 2.26, "results_file": "held_out_results_bert_base.json"},
]

summary = []
for m in model_ladder:
    fpath = os.path.join(RESULTS_DIR, m["results_file"])
    if os.path.exists(fpath):
        with open(fpath) as f:
            data = json.load(f)
        m["macro_f1"] = data["macro_f1"]
    else:
        m["macro_f1"] = "PENDING — run Activity 20 for this model"
    summary.append(m)

with open(f"{OUT_DIR}/size_accuracy_tradeoff.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Size-accuracy tradeoff table:")
for m in summary:
    print(f"  {m['model']:12s} params={m['params_M']}M  mem={m['memory_MB']}MB  "
          f"loss={m['train_loss']}  macro_f1={m['macro_f1']}")

print("\nActivity 21 complete. Add real eval results to complete the analysis.")
