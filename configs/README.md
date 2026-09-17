# `configs/` — all configuration lives here

The original research scripts had settings (dataset paths, checkpoint paths,
hyperparameters, thresholds) hardcoded as constants at the top of each file. In
the product package those move **here**, as YAML, so nothing environment-specific
is baked into the code and no local absolute paths are ever committed.

| Folder | Configures |
|--------|-----------|
| `data/`  | Where each dataset lives and how it's loaded (one file per dataset: Turkey, OSU, Chile). |
| `model/` | Which backbone + adaptation head, and which pinned weights (one file per method). |
| `audit/` | Drift thresholds and masking parameters — the numbers the gate uses to decide "trustworthy or not." |
| `serve/` | The service config: which model version to load, gate thresholds, host/port. |

## Convention

- Files ending in **`.example.yaml`** are committed templates. Copy to the same
  name without `.example` (e.g. `api.example.yaml` → `api.yaml`) and fill in local
  values. The non-example files are **gitignored** so real paths never leave your
  machine.
- If you add a new tunable value to the code, add it here too. Reviewers should be
  able to see every knob in `configs/` without reading the source.
