.PHONY: install install-dev test lint fetch-fred train dashboard clean help

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

test-quick: ## Run tests without optional-dependency tests
	$(PYTHON) -m pytest tests/ -v -k "not linearmodels"

# ── Data ─────────────────────────────────────────────────────────────────
fetch-fred: ## Fetch FRED series (requires FRED_API_KEY env var)
	$(PYTHON) scripts/fetch_fred.py

fetch-cms: ## Fetch CMS health data (no API key needed)
	$(PYTHON) -c "from src.health_data import fetch_cms_dataset; print('CMS fetch ready')"

# ── Train ────────────────────────────────────────────────────────────────
train: ## Train recession classifier
	$(PYTHON) scripts/train_recession_classifier.py

# ── Apps ─────────────────────────────────────────────────────────────────
dashboard: ## Launch Streamlit dashboard
	streamlit run apps/recession_dashboard.py

# ── Housekeeping ─────────────────────────────────────────────────────────
clean: ## Remove cached data and Python artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
