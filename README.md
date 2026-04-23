# Multi-agent auction and bargaining experiments

This repo keeps the runnable auction, bargaining, internal-market, zoning, escrow, and standards experiment code. The only saved initiative benchmark artifacts we still keep are the authoritative escrow and standards outputs.

## Install
Requires Python 3.11+.

```bash
pip install -e .
```

Set `OPENAI_API_KEY` and `OPENAI_BASE_URL` if needed. To use Gemini, set `GEMINI_API_KEY` or `GOOGLE_API_KEY` and pass `--provider gemini --model <gemini-model>`.

## Core runs
Dry run:

```bash
python -m experiments.runner --mode vickrey --condition all --seeds 1 --dry-run
```

Vickrey example:

```bash
python -m experiments.runner --mode vickrey --condition collusion_channel --seeds 3
```

Shapley example:

```bash
python -m experiments.runner --mode shapley --condition dm_late_reveal --seeds 2
```

These runs write fresh output to runner-specific directories when you execute them.

## Kept benchmark surfaces
Escrow benchmark code lives in `experiments/escrow_market/`. The kept saved output and plots are:

- `initiative_benchmarks/escrow_inspection/runs/authoritative/escrow_market_results.jsonl`
- `initiative_benchmarks/escrow_inspection/plots/escrow_market_fulfillment_rate_by_arm_model.png`
- `initiative_benchmarks/escrow_inspection/plots/escrow_market_institution_activation_rate_by_arm_model.png`

Standards benchmark code lives in `experiments/standards_market/`. The kept saved output and note are:

- `initiative_benchmarks/standards_market/runs/authoritative/standards_market_results.jsonl`
- `initiative_benchmarks/standards_market/notes/autarkic_localism.md`
- `initiative_benchmarks/standards_market/plots/autarkic_localism_summary.svg`

## Re-run kept benchmarks
Escrow smoke run to a temporary folder outside the repo:

```bash
python -m experiments.escrow_market.runner --provider codex --models gpt-5.4,gpt-5.2 --seeds 1 --firms 6 --rounds 3 --codex-timeout 600 --output-dir /tmp/llm-auctions-bargaining-escrow-smoke
```

Standards smoke run to a temporary folder outside the repo:

```bash
python -m experiments.standards_market.runner --provider codex --models gpt-5.4 --seeds 1 --firms 4 --rounds 3 --deadline-round 3 --codex-timeout 600 --output-dir /tmp/llm-auctions-bargaining-standards-smoke
```
