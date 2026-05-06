.PHONY: install install-dev test lint fetch-fred clean help

PYTHON ?= python

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────────────────────
install: ## Install runtime dependencies
	pip install -r requirements.txt

install-dev: install ## Install runtime + dev dependencies
	pip install -r requirements-dev.txt

# ── Quality ──────────────────────────────────────────────────────────────
test: ## Run test suite
	$(PYTHON) -m pytest tests/ -v

# ── Data ─────────────────────────────────────────────────────────────────
fetch-fred: ## Fetch FRED series (requires FRED_API_KEY env var)
	$(PYTHON) scripts/fetch_fred.py --config configs/fred.yaml

# ── Housekeeping ─────────────────────────────────────────────────────────
clean: ## Remove cached data and Python artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
