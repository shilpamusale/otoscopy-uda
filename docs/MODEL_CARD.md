# Model Card

<!-- TODO (Jordan + Ishi): complete before any external sharing. -->

## Intended use
Research/pilot cross-site otoscopy (Normal vs. Abnormal) classification, served
**behind an audit gate** (see docs/adr/0001). Not a cleared diagnostic device.

## Model
- Backbone: ResNet50, **pinned** ImageNet weights (not the DEFAULT alias).
- Adaptation head: DANN / MCC / CORAL (per checkpoint).

## Training data
Three clinical sites — Turkey (source), OSU (target), Chile (target). See
docs/DATA_CARD.md.

## Limitations (the important part)
- Cross-site performance can drop substantially; accuracy on one site does not
  transfer. The audit gate exists precisely because of this.
- May rely on peripheral, non-diagnostic features; the masking score surfaces this.
- Evaluation used balanced 50/50 test sets; prevalence-adjusted performance differs.

## Ethical / regulatory
See docs/DEPLOYMENT.md — SaMD, HIPAA/PHI, and clinical validation are out of
scope for the current stage.
