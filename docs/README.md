# `docs/` — project documentation

Reference documents for anyone using, reviewing, or deploying this project.

| File / folder | What it is |
|---------------|-----------|
| `MODEL_CARD.md` | The model's intended use, training data, performance, and — importantly — its **cross-site limitations**. The honest "when does this work / when does it not." |
| `DATA_CARD.md`  | Dataset provenance for all three sites and the **non-redistribution** notice ("publicly available" ≠ "redistributable"). |
| `DEPLOYMENT.md` | The honest deployment posture: what a real hospital deployment would require beyond this repo (regulatory/SaMD, HIPAA/PHI, DICOM, validation). **Read before assuming anything here is production-ready.** |
| `adr/`          | **Architecture Decision Records** — short notes capturing *why* a significant choice was made, so it doesn't get re-litigated or silently reversed. |

## Why ADRs?

Several methodological choices in this project are the kind reviewers ask about
(joint vs. per-dataset t-SNE, shared MMD bandwidth, mask fill method, the
audit-gated architecture itself). An ADR records the decision and its rationale
in one short file. See `adr/README.md`.
