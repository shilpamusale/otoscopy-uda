# `src/` — the installable product package

This is the **product** side of the repo. Everything here is refactored,
configurable, importable, and tested — as opposed to `experiments/`, which holds
the frozen research scripts.

There is one package: **`otoscopy_audit/`**. Install it with `pip install -e .`
from the repo root and import it as `import otoscopy_audit`.

**Rule of thumb for contributors:** if code needs to run in the deployed service,
be imported by another module, or be covered by a test, it belongs here — not in
`experiments/`. New configurable values (paths, thresholds, hyperparameters) go
in `configs/`, never hardcoded in these files.
