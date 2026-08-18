"""
Activity 15 — Organize Existing Evaluation Resources
=====================================================
INSTRUCTIONS FOR PROFESSOR/TA:
Place the following files in /workspace/eir-project/data/evaluation/:
  1. student_responses.csv   — columns: response_id, response_text
  2. rubrics.csv             — columns: rubric_id, category, label, definition, keywords
  3. human_annotations.csv  — columns: response_id, rubric_id, category, label (present/absent)

Then run: python3 activity15_organize_evaluation.py
"""

import pandas as pd
import json
import os
from pathlib import Path

EVAL_DIR = "/workspace/eir-project/data/evaluation"
OUT_DIR  = "/workspace/eir-project/data/evaluation/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# ── LOAD FILES (drop your files in EVAL_DIR) ──────────────────
responses   = pd.read_csv(f"{EVAL_DIR}/student_responses.csv")
rubrics     = pd.read_csv(f"{EVAL_DIR}/rubrics.csv")
annotations = pd.read_csv(f"{EVAL_DIR}/human_annotations.csv")

print(f"Responses:   {len(responses)}")
print(f"Rubrics:     {len(rubrics)}")
print(f"Annotations: {len(annotations)}")

# ── ASSIGN STABLE IDs ─────────────────────────────────────────
responses["anon_id"] = ["R{:04d}".format(i) for i in range(len(responses))]

# ── ALIGN ─────────────────────────────────────────────────────
aligned = annotations.merge(responses, on="response_id") \
                      .merge(rubrics, on=["rubric_id","category"])

# ── VALIDATE ──────────────────────────────────────────────────
assert aligned["label"].isin(["present","absent"]).all(), "Invalid labels found"
assert aligned["anon_id"].nunique() == len(responses), "ID mismatch"

# ── SAVE ──────────────────────────────────────────────────────
aligned.to_csv(f"{OUT_DIR}/aligned_evaluation.csv", index=False)
rubrics.to_json(f"{OUT_DIR}/rubrics.json", orient="records", indent=2)
annotations[["anon_id","category","label"]].to_csv(
    f"{OUT_DIR}/human_reference_labels.csv", index=False)

print(f"\nSaved to {OUT_DIR}")
print("Activity 15 complete.")
