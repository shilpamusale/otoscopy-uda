# Deployment posture (read this before assuming production-readiness)

This repository is a **portfolio / pilot-stage** artifact, deliberately
*structured* like a deployable product. It is **not** a cleared or validated
medical device, and nothing here should be used to inform patient care as-is.

The point of this document is to be explicit about the gap between "structured
like a product" and "safe to deploy in a hospital," so the scope is honest.

## What IS built (or scaffolded) here
- An installable package with a clean research→product separation.
- The **audit gate**: drift (MMD) and peripheral-reliance (masking) checks that
  decide whether a prediction can be trusted on a given site.
- A REST serving path (FastAPI) that routes every prediction through the gate.
- Config-driven paths/thresholds (no hardcoded local paths), tests, containerization.

## What a REAL hospital deployment would additionally require
1. **Regulatory assessment (SaMD).** Software that informs a clinical decision
   about a patient is likely Software as a Medical Device and may be FDA-regulated.
   This needs a proper regulatory pathway determination before any clinical use.
2. **HIPAA-compliant PHI handling.** Otoscopy images are PHI. A real deployment
   needs de-identification at ingestion (started in `data/deident.py`), access
   control, audit logging, encryption in transit and at rest, and a defined
   data-retention policy.
3. **DICOM/PACS integration.** Routine hospital workflow pushes DICOM to a PACS
   rather than uploading files to REST. See `src/otoscopy_audit/serve/dicom/`
   for the intended (unbuilt) design.
4. **Clinical validation.** Prospective, site-specific validation — exactly the
   cross-site generalization question the manuscript raises. The audit gate
   *supports* this but does not replace it.
5. **Monitoring & governance.** Ongoing drift monitoring in production, a model
   registry with versioning/rollback, and a process for revalidating after model
   or data changes.

## Summary
The audit-gated design is the honest core: it makes the model's trustworthiness
*measurable per site*. But measurability is a precondition for clinical use, not
clearance for it. Treat everything here as pilot infrastructure.
