PYTHONPATH := .
UV := uv

.PHONY: setup check test llm-live llm-live-escrow llm-live-standards deps-audit all

setup:
	PYTHONPATH=$(PYTHONPATH) $(UV) run python -c "import experiments"

check:
	PYTHONPATH=$(PYTHONPATH) $(UV) run --with ruff ruff check experiments analysis tests

test:
	PYTHONPATH=$(PYTHONPATH) $(UV) run --with pytest pytest -q

llm-live: llm-live-escrow llm-live-standards

llm-live-escrow:
	PYTHONPATH=$(PYTHONPATH) $(UV) run python -m experiments.escrow_market.runner --provider codex --models gpt-5.4,gpt-5.2 --seeds 1 --firms 6 --rounds 3 --codex-timeout 600 --output-dir /tmp/llm-auctions-bargaining-escrow-smoke

llm-live-standards:
	PYTHONPATH=$(PYTHONPATH) $(UV) run python -m experiments.standards_market.runner --provider codex --models gpt-5.4 --seeds 1 --firms 4 --rounds 3 --deadline-round 3 --codex-timeout 600 --output-dir /tmp/llm-auctions-bargaining-standards-smoke

deps-audit:
	PYTHONPATH=$(PYTHONPATH) $(UV) run --with pip-audit pip-audit

all: check test
