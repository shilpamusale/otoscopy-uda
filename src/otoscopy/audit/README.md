# `audit/` — the product differentiator

This is the heart of the repo. Our manuscript found that accuracy overstates
cross-site reliability and that models may lean on peripheral features. This
package turns those *findings* into runtime *checks*.

| Module | What it does | Reused from |
|--------|--------------|-------------|
| `drift.py`   | Computes **MMD** between an incoming batch and the training distribution. High MMD = the new site's images have drifted; predictions shouldn't be trusted. | The MMD computation in the analysis scripts. |
| `masking.py` | Scores how much a prediction relies on the **periphery** vs. the eardrum itself, by masking and measuring the performance drop. | `PeripheralMasking_FINAL.py`. |
| `gate.py`    | The decision layer. Combines the drift and masking signals and decides: **allow** the prediction, **warn**, or **withhold**. | New — this is the product logic. |

## Why the review bugfixes are load-bearing here

Two issues we found in review stop being "methods nitpicks" and become
correctness bugs in a live monitor:

- **MMD bandwidth** must be **shared** across the before/after (or train/incoming)
  comparison. A monitor whose bandwidth drifts isn't measuring drift — it's
  measuring its own hyperparameter.
- **Masking sample counts** must match across the masked/original arms. The
  original `continue`-inside-`if masked:` bug silently dropped images from one
  arm; in a scoring context that produces a meaningless score.

Fix these *here*, with tests in `tests/test_drift.py` and
`tests/test_masking_geometry.py`.
