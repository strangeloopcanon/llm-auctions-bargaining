PYTHONPATH := .
UV := uv

.PHONY: setup check test llm-live deps-audit all

setup:
	PYTHONPATH=$(PYTHONPATH) $(UV) run python -c "import experiments"

check:
	PYTHONPATH=$(PYTHONPATH) $(UV) run --with ruff ruff check experiments analysis tests

test:
	PYTHONPATH=$(PYTHONPATH) $(UV) run --with pytest pytest -q

llm-live:
	PYTHONPATH=$(PYTHONPATH) $(UV) run python -m experiments.ip_completion_market.runner --provider codex --models gpt-5.4,gpt-5.2 --seeds 1 --firms 4 --modules 4 --orders 2 --rounds 1 --codex-timeout 300 --output-dir runs_ip_completion_market/live_smoke

deps-audit:
	PYTHONPATH=$(PYTHONPATH) $(UV) run --with pip-audit pip-audit

all: check test
