"""The audit layer — the product differentiator.

Turns the manuscript's findings into runtime checks:
    drift   — MMD between incoming images and the training distribution
    masking — how much a prediction relies on the (non-diagnostic) periphery
    gate    — combines the above and decides: allow / warn / withhold

This is what separates the service from a naive classifier API. See the
subpackage README for how the review bugfixes (shared MMD bandwidth, matched
masking sample counts) become correctness requirements here.
"""
