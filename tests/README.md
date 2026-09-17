# `tests/` — unit and regression tests

Run everything with `make test` (or `pytest`) from the repo root.

## What's here

- **`fixtures/`** — a tiny (~20-image) **synthetic** dataset so tests run fast and
  commit no real patient data. Same `Normal/`/`Abnormal/` layout as a real
  dataset.
- **`test_masking_geometry.py`** — regression test for the **sample-count
  divergence bug** (the `continue` inside `if masked:` that dropped images from
  one arm). Guards the masking scorer against silently unpairing its arms.
- **`test_drift.py`** — checks the MMD drift computation is deterministic and that
  the **bandwidth is shared** across the comparison (the other review finding).
- **`test_gate.py`** — checks the gate **blocks** clearly out-of-distribution or
  periphery-reliant inputs and **allows** clean in-distribution ones.

## Convention

New product code in `src/` should come with a test here. The two bug-regression
tests exist specifically so those issues can't silently return.
