"""Command-line entry point for otoscopy_audit.

Example:
    python -m otoscopy_audit.cli audit --ckpt path/to/model.pth --target-dir path/to/images

Subcommands (planned):
    audit  — run drift + masking + gate on a folder of new-site images and print
             a report (does the model hold up on this site?)
    serve  — launch the FastAPI service (thin wrapper around serve.api)
"""
from __future__ import annotations


def main() -> int:
    """Parse arguments and dispatch to a subcommand.

    TODO: implement the `audit` and `serve` subcommands.
    """
    raise NotImplementedError("cli.main is a stub — implement alongside the audit layer.")


if __name__ == "__main__":
    raise SystemExit(main())