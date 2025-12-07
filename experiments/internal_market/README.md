# Internal market experiment (firm vs market boundary)

Implements a Coase/Hayek-style internal capital market with four LLM department agents (Marketing, Sales, Product, Engineering). Each “quarter” they spend points to buy engineering capacity for features; selection uses points-per-eng-cost; payoffs update future budgets. Outputs go to `runs_internal_market/`.

## Run
Dry run (no model calls):
```
python -m experiments.internal_market.runner --dry-run
```

Live with GPT-5.1:
```
python -m experiments.internal_market.runner --seeds 3 --quarters 10 --features 20 --provider openai --model gpt-5.1
```

## Mechanics
- Features: generated with hidden ground-truth values per dept and eng cost.
- Departments: role-specific system prompts; budgets update from utility (selected feature value + thrift on unspent points).
- Allocator: greedy ratio of total support / eng cost under capacity.
- Schema: agents must return JSON `{"commentary": str, "bids": [{"feature_id": str, "points": int}]}` with budget respected.
- Metrics logged: bids, support, selected features, utilities, budgets per quarter, final summaries.
