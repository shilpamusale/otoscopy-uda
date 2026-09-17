# `experiments/` — research code (FROZEN)

This is the **research** side of the repo. The scripts here are the exact code
that produced the manuscript's results. They are kept **verbatim and frozen**:
we do **not** refactor them, so that anyone can reproduce the paper.

If you want to change how something works for the *product*, do it in
`src/otoscopy_audit/` — not here. This directory is a reproducibility record, not
a place for ongoing development.

## Contents

- **`manuscript_2026/`** — the scripts behind the current manuscript.

## Why keep these at all?

Two reasons:
1. **Reproducibility.** A reviewer or a future reader can run these and get the
   paper's numbers, unmodified.
2. **The portfolio story.** The contrast between these scripts and the refactored
   `src/` package *is* the research→product transition. Deleting them would erase
   the most valuable part of the narrative.

> Note: these scripts contain hardcoded local paths and known issues we
> identified in review (see `docs/` and the manuscript review). That's expected —
> they're a snapshot of the research as it was. The fixes live in `src/`.
