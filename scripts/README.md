# `scripts/` — operational helper scripts

Standalone utilities for running and maintaining the project. Unlike
`experiments/` (frozen research) and `src/` (the product package), these are
small operational tools.

- **`download_data.py`** — verifies a dataset is present with the expected
  `Normal/`/`Abnormal/` layout, and is the intended home for per-dataset
  download/checksum logic. It **bundles no images** — data is obtained from each
  original source under its own terms (see `docs/DATA_CARD.md`).

  ```bash
  python scripts/download_data.py --dataset OSU --dest /path/to/OSUdataset --verify-only
  ```

Add future operational tooling here (e.g. checkpoint export, batch audit runs).
