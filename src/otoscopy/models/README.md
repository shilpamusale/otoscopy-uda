# `models/` — backbone, adaptation heads, checkpoint loading

The neural-network definitions and how trained weights are loaded.

Expected contents:
- **`backbone.py`** — the ResNet50 feature extractor. **Weights are pinned to an
  explicit `ResNet50_Weights` version**, not the `DEFAULT` alias. (The original
  scripts used `DEFAULT`, which silently resolves to whatever the installed
  torchvision ships — a reproducibility risk we flagged in review.)
- **adaptation heads** — DANN, MCC, and CORAL, sharing a common interface so the
  rest of the code doesn't care which method produced a checkpoint.
- **`registry.py`** — loads a checkpoint by model **version/name** and records
  which method + config produced it, so a deployed model is always traceable.

**Contributor note:** any new method should implement the shared head interface
so `serve/` and `audit/` can treat it uniformly.
