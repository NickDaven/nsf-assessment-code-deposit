"""
Activity 22 — Package the Final Model
======================================
Packages the best model (distilbert-triplet-statics) with everything
needed for local inference: tokenizer, weights, rubrics, thresholds,
inference script, and model card.
"""
import json, os, shutil, hashlib
from datetime import datetime

OUT_DIR = "/workspace/eir-project/final_package"
os.makedirs(OUT_DIR, exist_ok=True)

# Copy best model checkpoint
src = "/workspace/eir-project/data/checkpoints/distilbert-triplet-statics"
dst = f"{OUT_DIR}/model"
if os.path.exists(src):
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"Copied model to {dst}")

# Copy inference script
shutil.copy("/workspace/eir-project/src/activity18_inference_pipeline.py",
            f"{OUT_DIR}/inference.py")

# Write model card
model_card = """# EiR Domain-Adapted Statics Model

## Model Details
- Base model: distilbert-base-uncased
- Domain: Engineering Statics (free-body diagrams, equilibrium, support reactions)
- Training: Corpus-based triplet training on 14 ASEE statics pedagogy papers
- Corpus version: v1.0

## Intended Use
Rubrics-based automatic assessment of student answers in engineering statics.
Compare student response units against rubric indicator embeddings using cosine similarity.

## Usage
```python
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("./model")
response_emb = model.encode("The free body diagram shows forces in equilibrium")
rubric_emb   = model.encode("Student correctly identifies all forces")
score = util.cos_sim(response_emb, rubric_emb).item()
```

## Limitations
- Trained only on statics domain (14 papers, ~6800 sentences)
- Learning-process domain corpus is in progress
- Thresholds require calibration on real annotated student responses

## Placeholder Files Needed
Place in final_package/evaluation/:
- student_responses.csv
- rubrics.json
- human_annotations.csv

## Training Details
- Triplet loss, 3 epochs, batch size 16, lr 5e-5, seed 42
- Proximity and thematic triplets from corpus
- Final triplet loss: 1.38
"""

with open(f"{OUT_DIR}/MODEL_CARD.md", "w") as f:
    f.write(model_card)

# Write reproduction guide
guide = """# Reproduction Guide

## Environment
pip install transformers torch sentence-transformers datasets spacy accelerate
python3 -m spacy download en_core_web_sm

## Steps to Reproduce
1. Run corpus collection: UWSS discovery (OpenAlex, arXiv)
2. Run extraction: PyMuPDF on downloaded PDFs
3. Run cleaning: activity4_clean_segment.py
4. Run training data creation: activity5_create_training_data.py
5. Run triplet training: activity11_triplet_training.py
6. Run inference: python3 inference.py

## Corpus Version
v1.0 — SHA-256 manifest at data/corpus_manifest_v1.json

## GitHub
https://github.com/HrijP/eir-project
"""

with open(f"{OUT_DIR}/REPRODUCTION_GUIDE.md", "w") as f:
    f.write(guide)

# Checksums
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

checksums = {}
for root, dirs, files in os.walk(OUT_DIR):
    for fname in files:
        fpath = os.path.join(root, fname)
        rel   = os.path.relpath(fpath, OUT_DIR)
        checksums[rel] = sha256(fpath)

with open(f"{OUT_DIR}/checksums.json", "w") as f:
    json.dump(checksums, f, indent=2)

print(f"Package created at {OUT_DIR}")
print(f"Files: {len(checksums)}")
print("Activity 22 complete.")
