# Makefile — common developer commands. Run `make help` for the list.
# Thin wrappers so contributors don't memorize invocations.

.PHONY: help install test lint serve audit

help:
	@echo "install  - editable install with dev tools"
	@echo "test     - run the test suite (pytest)"
	@echo "lint     - run ruff"
	@echo "serve    - run the audit-gated API locally (uvicorn, reload)"
	@echo "audit    - run the audit CLI (pass ARGS='--ckpt ... --target-dir ...')"

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

serve:
	uvicorn otoscopy_audit.serve.api:app --reload --port 8000

audit:
	python -m otoscopy_audit.cli audit $(ARGS)
