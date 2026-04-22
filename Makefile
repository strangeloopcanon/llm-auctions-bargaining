PYTHONPATH := .
UV := uv

.PHONY: setup check test llm-live llm-live-completion llm-live-escrow llm-live-certification deps-audit all

setup:
	PYTHONPATH=$(PYTHONPATH) $(UV) run python -c "import experiments"

check:
	PYTHONPATH=$(PYTHONPATH) $(UV) run --with ruff ruff check experiments analysis tests

test:
	PYTHONPATH=$(PYTHONPATH) $(UV) run --with pytest pytest -q

llm-live:
	PYTHONPATH=$(PYTHONPATH) $(UV) run python -m experiments.ip_brokered_market.runner --provider codex --models gpt-5.4,gpt-5.2 --seeds 1 --firms 4 --modules 4 --orders 2 --rounds 3 --codex-timeout 600 --output-dir runs_ip_brokered_market/live_smoke

llm-live-completion:
	PYTHONPATH=$(PYTHONPATH) $(UV) run python -m experiments.ip_completion_market.runner --provider codex --models gpt-5.4,gpt-5.2 --seeds 1 --firms 4 --modules 4 --orders 2 --rounds 1 --scenarios partner_directory --codex-timeout 300 --output-dir runs_ip_completion_market/live_smoke

llm-live-escrow:
	PYTHONPATH=$(PYTHONPATH) $(UV) run python -m experiments.escrow_market.runner --provider codex --models gpt-5.4,gpt-5.2 --seeds 1 --firms 6 --rounds 3 --codex-timeout 600 --output-dir runs_escrow_market/live_smoke

llm-live-certification:
	PYTHONPATH=$(PYTHONPATH) $(UV) run python -m experiments.certification_market.runner --provider codex --models gpt-5.4,gpt-5.2 --seeds 1 --firms 4 --rounds 4 --deadline-round 4 --certification-capacity 2 --shipping-capacity 2 --codex-timeout 600 --output-dir runs_certification_market/live_smoke

deps-audit:
	PYTHONPATH=$(PYTHONPATH) $(UV) run --with pip-audit pip-audit

all: check test
