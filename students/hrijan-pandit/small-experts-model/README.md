# Small Domain-Aware Language Models for Rubrics-Based Automatic Assessment

## Overview
This repository contains the complete corpus processing, model training, evaluation, and inference pipeline for developing small domain-aware language models for rubrics-based automatic assessment of student responses in two domains: engineering statics and learning processes.

## Repository Structure

small-experts-model/
├── src/
│ ├── activity15_organize_evaluation.py — align student responses, rubrics, annotations
│ ├── activity16_create_splits.py — calibration/held-out splits
│ ├── activity18_inference_pipeline.py — common rubric-assessment inference
│ ├── activity19_calibrate_thresholds.py — category-specific threshold calibration
│ ├── activity20_held_out_evaluation.py — human-machine comparison
│ ├── activity21_error_analysis.py — error analysis and size-accuracy tradeoff
│ └── activity22_package_model.py — final model packaging
├── experiments/
│ └── registry.json — all experiment logs
├── config.yaml — UWSS corpus discovery configuration
├── MODEL_CARD.md — model details and usage
├── REPRODUCTION_GUIDE.md — step-by-step reproduction instructions
└── requirements_lock.txt — exact package versions


## Installation
```bash
pip install transformers torch sentence-transformers datasets spacy pandas pymupdf accelerate google-cloud-storage
python3 -m spacy download en_core_web_sm
```

## Corpus
- **Source tool**: Universal Web Scraping System (UWSS) — https://github.com/duynguyenxc/Universal-Web-Scraping-System-high-level-update
- **Statics domain**: 14 ASEE engineering-education papers (OpenAlex discovery)
- **Text extraction**: PyMuPDF
- **Sentences**: ~6,800 after cleaning and segmentation
- **Corpus version**: v1.0 (SHA-256 manifest available in GCS: gs://eir-research-corpus/)

## Models Trained
| Model | Parameters | MLM Loss | Triplet Loss | Checkpoint |
|---|---|---|---|---|
| TinyBERT_General_4L_312D | 14.4M | 5.38 | — | tinybert-statics |
| distilbert-base-uncased | 66.4M | 2.36 | 1.38 | distilbert-triplet-statics |
| bert-base-uncased | 109.5M | 2.26 | — | bert-base-statics |
| TinyBERT (distilled) | 14.4M | — | 0.109 | tinybert-distilled-statics |

## Training Settings
- Epochs: 3 (MLM), 3 (triplet), 2 (ablations)
- Batch size: 16
- Learning rate: 5e-5
- Max sequence length: 128
- MLM masking probability: 15%
- Random seed: 42
- Hardware: Vast.ai RTX 5090 (32GB VRAM)
- Runtime: 17-97 seconds per model

## Evaluation Status
Activities 15-21 (human-machine evaluation) require student response data, rubrics, and human annotations from the project supervisor. Placeholder scripts are fully built and ready to run once files are provided.

## Cloud Storage
- Platform: Google Cloud Storage
- Bucket: gs://eir-research-corpus/ (us-central1, private)
- Contents: corpus files, training datasets, model checkpoints

## GitHub
- Code: https://github.com/HrijP/eir-project
- NSF Deposit: https://github.com/NickDaven/nsf-assessment-code-deposit/tree/main/students/hrijan-pandit
