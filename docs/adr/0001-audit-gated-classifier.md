# 0001 — Audit-gated classifier (not a bare classifier API)

**Status:** accepted

## Context
The manuscript's core finding is that standard accuracy overstates cross-site
reliability and that models may rely on non-diagnostic peripheral features.
Serving the classifier directly would ship a product our own research says is
not trustworthy on a new site.

## Decision
The classifier is served **behind an audit gate**. Every prediction request runs
a distribution-drift check (MMD) and a peripheral-reliance check (masking) first;
the gate then decides to allow, warn, or withhold the prediction. The gate policy
lives in `otoscopy_audit.audit.gate` and no serving path bypasses it.

## Consequences
- The product's value proposition is honesty about when it can be trusted — a
  differentiator, not just a wrapper.
- Two review findings become correctness requirements (see ADR 0002 and the
  masking sample-count test), not optional cleanups.
- Slightly more compute per request (the audit checks). Acceptable for the
  intended use.
