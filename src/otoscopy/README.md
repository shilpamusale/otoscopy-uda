# `otoscopy_audit/` — the package

The audit-gated otoscopy classifier, as a real Python package. Submodules:

| Subpackage | Responsibility |
|------------|---------------|
| `data/`   | Loading images, the dataset registry, download + checksum, and **PHI de-identification** at ingestion. |
| `models/` | The ResNet50 backbone, the DANN/MCC/CORAL adaptation heads, and checkpoint/version loading. |
| `audit/`  | **The differentiator.** Drift detection (MMD), peripheral-reliance scoring (masking), and the trust **gate**. |
| `serve/`  | The FastAPI service and (later) the DICOM/PACS listener. |
| `cli.py`  | Command-line entry point: `python -m otoscopy_audit.cli audit --ckpt ... --target-dir ...`. |

## How a request flows

```
image → data.deident → audit.drift (in-distribution?) 
                      → audit.masking (leaning on periphery?)
                      → audit.gate (trustworthy here? → allow / warn / withhold)
                      → models (prediction, only if the gate allows)
```

The `audit/` layer is what separates this from a naive classifier API. Read
`audit/README.md` first — it's the heart of the project.
