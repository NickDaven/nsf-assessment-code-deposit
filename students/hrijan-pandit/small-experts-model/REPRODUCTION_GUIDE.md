# Reproduction Guide

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
