# 0002 — Shared MMD bandwidth across the comparison

**Status:** accepted

## Context
MMD with an RBF kernel depends on a bandwidth (gamma). An earlier version
estimated gamma **separately** for the two sides being compared. When the two
sides use different bandwidths, the statistic partly reflects the bandwidth
choice rather than the actual distribution difference — a confound in the
manuscript and an outright bug in a live drift monitor.

## Decision
Use a **single, fixed bandwidth** (configured in `configs/audit/`) for both sides
of every MMD comparison. `otoscopy_audit.audit.drift.compute_mmd` takes gamma as
a required argument; it never estimates it per input.

## Consequences
- Drift values are comparable across batches and over time — a prerequisite for
  thresholding in the gate.
- The bandwidth becomes a config decision to document and justify, not an
  implementation detail. Covered by `tests/test_drift.py`.
