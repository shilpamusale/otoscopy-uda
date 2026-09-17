# `manuscript_2026/` — scripts behind the UDA manuscript

The original analysis and training scripts, unmodified. Each is a standalone,
top-to-bottom script (not an importable module) with its settings in a
`USER SETTINGS` block at the top.

| Script | Role |
|--------|------|
| `DANN_Resnet50.py` | Train the DANN (domain-adversarial) adaptation model. |
| `MCC_Resnet50.py` | Train the MCC (minimum class confusion) model. |
| `CORAL_Resnet50.py` | Train the CORAL (correlation alignment) model. |
| `BatchNorm_Resnet50.py` | BatchNorm-adaptation baseline. |
| `ADDA.py` | ADDA baseline. |
| `DANN_FullFeatureAnalysis_FINAL.py` | Feature extraction, t-SNE, linear probes, MMD, and the statistical tests. |
| `LogisticRegression_Features_DANN.py` | Logistic-regression probes on extracted features. |
| `PeripheralMasking_FINAL.py` | Peripheral-masking evaluation (annotate, then 30-repeat masked-vs-original). |

**Do not edit these to fix bugs or change behavior.** They are the frozen record.
The refactored, corrected versions of this logic live in `src/otoscopy_audit/`
(training/feature code in `models/` and `data/`, the MMD and masking code in
`audit/`).
