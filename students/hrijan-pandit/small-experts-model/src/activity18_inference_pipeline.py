"""
Activity 18 — Common Machine Rubric-Assessment Pipeline
========================================================
This script runs inference for ANY trained model against rubric indicators.

USAGE:
  python3 activity18_inference_pipeline.py \
    --model_path /workspace/eir-project/data/checkpoints/distilbert-triplet-statics \
    --responses_file /workspace/eir-project/data/evaluation/splits/held_out_split.csv \
    --rubrics_file /workspace/eir-project/data/evaluation/processed/rubrics.json \
    --output_file /workspace/eir-project/data/results/distilbert_predictions.json

NOTE: Until real evaluation files are available, the script runs on
      placeholder data to verify the pipeline works end-to-end.
"""

import argparse, json, os
import pandas as pd
from sentence_transformers import SentenceTransformer, util

def make_response_units(text):
    """Create 1, 2, and 3 sentence response units."""
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 10]
    units = []
    for i in range(len(sentences)):
        units.append(sentences[i])
        if i+1 < len(sentences):
            units.append(sentences[i] + ". " + sentences[i+1])
        if i+2 < len(sentences):
            units.append(sentences[i] + ". " + sentences[i+1] + ". " + sentences[i+2])
    return units

def run_inference(model_path, responses, rubric_indicators, output_file):
    print(f"Loading model: {model_path}")
    model = SentenceTransformer(model_path)

    rubric_embeddings = model.encode(rubric_indicators, convert_to_tensor=True)

    results = []
    for resp_id, response_text in responses:
        units = make_response_units(response_text)
        if not units:
            units = [response_text]

        unit_embeddings = model.encode(units, convert_to_tensor=True)
        scores = util.cos_sim(unit_embeddings, rubric_embeddings)

        best_scores = []
        for r_idx, rubric in enumerate(rubric_indicators):
            col_scores = scores[:, r_idx].tolist()
            best_unit_idx = col_scores.index(max(col_scores))
            best_scores.append({
                "rubric": rubric,
                "best_score": round(max(col_scores), 4),
                "best_unit": units[best_unit_idx]
            })

        results.append({
            "response_id": resp_id,
            "response": response_text[:150],
            "rubric_scores": best_scores
        })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} predictions to {output_file}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--responses_file", default=None)
    parser.add_argument("--rubrics_file", default=None)
    parser.add_argument("--output_file", default="/workspace/eir-project/data/results/test_predictions.json")
    args = parser.parse_args()

    # Use placeholder data if real files not provided
    if args.responses_file and os.path.exists(args.responses_file):
        df = pd.read_csv(args.responses_file)
        responses = list(zip(df["response_id"], df["response_text"]))
    else:
        print("NOTE: Using placeholder responses — provide real responses_file when available")
        responses = [
            ("R001", "The free body diagram shows the weight force acting downward and normal force upward."),
            ("R002", "I drew all the forces but forgot to include friction."),
            ("R003", "Sum of forces equals zero so the body is in equilibrium."),
        ]

    if args.rubrics_file and os.path.exists(args.rubrics_file):
        with open(args.rubrics_file) as f:
            rubrics_data = json.load(f)
        rubric_indicators = [r["definition"] for r in rubrics_data]
    else:
        print("NOTE: Using placeholder rubrics — provide real rubrics_file when available")
        rubric_indicators = [
            "The student correctly identifies all forces acting on the body",
            "The free body diagram shows correct direction of forces",
            "Moment equilibrium is correctly applied",
            "Support reactions are correctly calculated",
        ]

    results = run_inference(args.model_path, responses, rubric_indicators, args.output_file)
    print(f"\nSample result: {json.dumps(results[0], indent=2)}")
    print("\nActivity 18 complete.")
