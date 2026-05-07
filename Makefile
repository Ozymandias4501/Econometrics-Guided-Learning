.PHONY: install install-dev test fetch-fred notebook clean help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────────────────────
install: ## Sync runtime dependencies (uv-managed venv)
	uv sync --no-dev

install-dev: ## Sync runtime + dev dependencies
	uv sync

# ── Quality ──────────────────────────────────────────────────────────────
test: ## Run test suite
	uv run pytest tests/ -v

# ── Data ─────────────────────────────────────────────────────────────────
fetch-fred: ## Fetch FRED series (reads FRED_API_KEY from environment)
	uv run python scripts/fetch_fred.py --config configs/fred.yaml

# ── Notebooks ────────────────────────────────────────────────────────────
notebook: ## Launch Jupyter in the uv-managed venv
	uv run jupyter notebook

# ── Housekeeping ─────────────────────────────────────────────────────────
clean: ## Remove cached data and Python artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
