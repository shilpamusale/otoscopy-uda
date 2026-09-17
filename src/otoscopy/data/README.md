# `data/` — datasets, loading, and de-identification

Everything about getting image data into the model **safely**.

Expected contents:
- **loaders** — turn a dataset directory (`Normal/`, `Abnormal/` subfolders) into
  batches, with the standard 224×224 ImageNet preprocessing.
- **registry** — a small map of known datasets (Turkey source, OSU target, Chile
  target) so code refers to them by name, not by path.
- **download + checksum** — verify a dataset is present and intact (see also
  `scripts/download_data.py`). **No images are ever committed to the repo.**
- **`deident.py`** — strips identifying metadata (e.g. EXIF, embedded tags) from
  images at ingestion. In a hospital context, otoscopy images are PHI; this is
  the first line of that handling. Minimal today, but real and load-bearing.

**Contributor note:** `deident.py` must run *before* an image is logged, cached,
or written anywhere. If you add a new ingestion path, route it through deident.
