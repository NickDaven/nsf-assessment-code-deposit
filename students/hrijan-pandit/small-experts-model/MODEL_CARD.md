# EiR Domain-Adapted Statics Model

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
