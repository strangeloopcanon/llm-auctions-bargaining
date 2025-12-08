# Multi-agent auction and bargaining experiments

Lightweight scaffolding to probe how language agents behave in Vickrey auctions and Shapley-style bargaining when they can talk, remember, and contest rules. Focus: cumulative effects of multiple agents coordinating or competing.

## Install
Requires Python 3.11+. Install deps:
```
pip install -e .
```
Set `OPENAI_API_KEY` (and `OPENAI_BASE_URL` if using a gateway). Default model: `gpt-5.1`.
To use Gemini, set `GEMINI_API_KEY` or `GOOGLE_API_KEY` and run with `--provider gemini --model <gemini-model>`.

## Run
Dry run (no model calls):
```
python -m experiments.runner --mode vickrey --condition all --seeds 1 --dry-run
```

Live run example (Vickrey, collusion arm, 3 seeds):
```
python -m experiments.runner --mode vickrey --condition collusion_channel --seeds 3
```

Live run with Gemini:
```
python -m experiments.runner --mode vickrey --condition collusion_channel --seeds 1 --provider gemini --model gemini-1.5-pro
```

Shapley bargaining with private DMs and late reveal of the reference split:
```
python -m experiments.runner --mode shapley --condition dm_late_reveal --seeds 2
```

Outputs land in `runs/` as JSONL logs per seed/condition plus a small summary JSONL.
For all experiments (including internal markets and IP licensing), see `RUNS_OVERVIEW.md` for a consolidated run log and analysis.

## Conditions (prioritized for institutional robustness)
- Vickrey: `collusion_channel` (one public message before bids), `memory_anchor` (3 repeated auctions with price history), `rule_challenge` (bidders can contest rules, explanations required), `explanation_coord` (messages + required rationale).
- Shapley: `dm_late_reveal` (private DMs allowed; mediator reveals reference split after round 1), `broadcast_never_reveal` (broadcast-only; no reference), `dm_never_reveal_adversarial` (private DMs; one contrarian player; no reference).

## Design notes
- Agents are given distinct personas (profit, fairness, compliance, cooperative, occasional contrarian). No heterogeneous model capabilities; all use GPT-5.1.
- Personas now include detailed 500+ word backgrounds covering objectives, risk attitudes, norms, and communication style to produce realistic four-years-out agent behavior.
- Vickrey side-channel messages provide a minimal coordination surface; memory arm tests anchoring across repeated auctions; rule-challenge arm logs attempts to renegotiate the mechanism.
- Shapley negotiation models coalition formation: optional private DMs, optional late release of a reference (Shapley-style) allocation, and an adversarial player variant to stress mediator robustness.
- All prompts force JSON output. Logs capture bids/proposals, messages, rule challenges, allocations, and distances to the reference split.

## Interpreting results
- Vickrey: watch bid compression under collusion, anchoring drift across rounds, and frequency/content of rule challenges; track allocative efficiency and revenue changes.
- Shapley: compare distance to the reference split with/without DMs and reference reveal; inspect coalition messages and whether agreements converge when the reference is withheld. Adversarial runs test whether a single actor derails convergence.

## Initial findings (early runs)
- Representative summaries and transcripts live in `runs/`. `RUNS_OVERVIEW.md` is the canonical log of all runs and results.
- Vickrey (early seeds 42–44): allocative efficiency held; side-channel messages reinforced truthful bidding; no rule challenges; overbids were only rounding effects; memory arm showed no drift.
- Shapley (early seeds 42–44): private DMs enabled coalition talk (e.g., p4/p5 coordinating premiums) and nudged allocations toward the reference, but most outcomes stayed materially off the Shapley-style split; broadcast-only often fell back to near-equal splits; no deadlocks or rule attacks observed.
