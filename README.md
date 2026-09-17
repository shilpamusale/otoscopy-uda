# otoscopy-uda

Cross-site robustness auditing for otoscopy (eardrum) image classification, with
an **audit-gated classifier** as the deployable product. Built from the research
codebase behind our unsupervised domain adaptation (UDA) manuscript.

> **For contributors (Jordan, start here):** every folder in this repo has its
> own `README.md` explaining what it's for — click into any directory to read it.
> Every source file starts with a header comment describing its role. This root
> README is the map; the folder READMEs are the detail.

---

## The one idea behind this repo

Our paper's central finding is that standard accuracy **overstates** how well
these models generalize to a new clinical site, and that they may lean on
non-diagnostic peripheral features. So the product here is **not** a bare
classifier API — that would be a product our own research says you shouldn't
trust.

Instead, the classifier sits **behind an audit gate**. For every image the
service:

1. checks whether the incoming image is **in-distribution** for the trained
   model (MMD drift check),
2. checks whether the prediction is **relying on the periphery** (masking score),
3. and only then decides whether to return the prediction, return it with a
   loud warning, or withhold it because the model can't be trusted on this site.

That gate is the direct operationalization of the manuscript's finding. It's
what makes this deployable *honestly*.

---

## Repository layout

| Path | What it is |
|------|-----------|
| **`src/otoscopy_audit/`** | The installable Python package — the **product** code (refactored, tested, configurable). |
| **`experiments/`** | The **research** code — original manuscript scripts, kept verbatim and frozen for reproducibility. |
| **`configs/`** | YAML configuration. Replaces the hardcoded local paths that were in the original scripts. |
| **`tests/`** | Unit + regression tests, including tests for the specific bugs we found during review. |
| **`docs/`** | Model card, data card, deployment notes, and architecture decision records (ADRs). |
| **`scripts/`** | Operational helper scripts (e.g. dataset download/verify). |
| `Dockerfile` / `docker-compose.yml` | Containerized serving. |
| `pyproject.toml` | Packaging, dependencies, lint/test config. |

### The two halves — why both exist

The split between `experiments/` and `src/` is deliberate and is the core of the
project:

- **`experiments/manuscript_2026/`** contains the exact scripts that produced the
  paper's results. They are **frozen** — we don't refactor them, so anyone can
  reproduce the manuscript.
- **`src/otoscopy_audit/`** is the same logic **refactored into a real package**:
  configurable, importable, tested, and deployable.

Keeping both, side by side, is the point. It shows the research to product
transition explicitly instead of overwriting one with the other.

---

## Quick start

```bash
# install (editable) + dev tools
pip install -e ".[dev]"

# copy the example config and set your local paths
cp configs/serve/api.example.yaml configs/serve/api.yaml

# run the tests
make test

# run the audit CLI against a checkpoint + a folder of new-site images
python -m otoscopy_audit.cli audit --ckpt path/to/model.pth --target-dir path/to/images

# serve the audit-gated API locally
make serve
```

---

## Status & scope

This is a **portfolio / pilot-stage** artifact that is *structured* like a
deployable product. It is **not** a cleared medical device. Before any real
hospital deployment, the additional requirements (regulatory assessment, a
HIPAA-compliant PHI pipeline, DICOM/PACS integration, clinical validation) are
documented honestly in docs/DEPLOYMENT.md. Read that before assuming anything
here is production-ready.

The repository is **private until the manuscript is published.**

## Contributing

See folder READMEs for where things go. In short: research code stays in
`experiments/` (frozen); new product code goes in `src/otoscopy_audit/` with a
test in `tests/`; new configurable values go in `configs/`, never hardcoded.
