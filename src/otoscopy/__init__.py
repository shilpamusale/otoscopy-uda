"""otoscopy_audit — audit-gated otoscopy classifier.

Product package built from the UDA research codebase. The classifier is served
*behind* a robustness audit gate: every prediction is checked for distribution
drift and peripheral-feature reliance before it is trusted.

Subpackages:
    data   — loading, dataset registry, download/checksum, PHI de-identification
    models — ResNet50 backbone, DANN/MCC/CORAL heads, checkpoint registry
    audit  — MMD drift detection, masking-based reliance score, and the gate
    serve  — FastAPI service (and, later, DICOM/PACS listener)

See the README in each subpackage for details.
"""

__version__ = "0.0.0"