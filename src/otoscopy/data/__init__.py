"""Data loading, dataset registry, and PHI de-identification.

Turns dataset directories (Normal/ + Abnormal/) into model-ready batches and
strips identifying metadata at ingestion. No image data is committed to the repo;
see scripts/download_data.py and docs/DATA_CARD.md.
"""