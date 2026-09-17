# Architecture Decision Records (ADRs)

Each ADR is a short, numbered, immutable note describing **one significant
decision**: the context, the choice, and the consequences. When a decision
changes, we don't edit the old ADR — we add a new one that supersedes it. This
gives a paper trail for the choices reviewers and future contributors will ask
about.

## Format

```
# NNNN — Short title
Status: proposed | accepted | superseded by NNNN
Context: what problem / question forced a decision
Decision: what we chose
Consequences: what follows (good and bad), what it rules out
```

## Index

- `0001-audit-gated-classifier.md` — why the classifier sits behind an audit gate
  rather than being served directly.
- `0002-shared-mmd-bandwidth.md` — why the MMD bandwidth is shared across the
  comparison instead of recomputed per condition.

Add new records as `NNNN-title.md`, incrementing the number.
