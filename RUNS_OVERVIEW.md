# Run log and prompts (LLM agent economics experiments)

This collects what we have run so far, the prompt scaffolding, and key takeaways for further study.

## Prompt scaffolding (where to find them)
- Vickrey / Shapley personas and schemas: `data/personas.json`, handlers in `experiments/envs/{vickrey,shapley}.py`, model plumbing in `experiments/agents.py`.
- Internal capital market (departments, bidding JSON schema, allocator): `experiments/internal_market/{prompts.py,simulation.py,allocator.py}` plus runner `experiments/internal_market/runner.py`.
- IP licensing market (firm strategy agent prompt, double auction matcher): `experiments/ip_market/{prompts.py,market.py,generator.py}`, runner `experiments/ip_market/runner.py`.
- Zoning Coase test prompts and mechanisms: `experiments/zoning/{prompts.py,simulation.py}` (not run in this round).

## Runs completed (OpenAI gpt-5.1 unless noted)

### Internal capital market (Coase/Hayek inside the firm)
- Seeds: 800, 801, 802; 10 quarters, 20 features. Outputs: `runs_internal_market/firm_market_seed{seed}_events.jsonl` (+ summaries) for the baseline regime.
- Observations:
  - GTM departments (Marketing, Sales) accumulate political capital; Engineering posts negative utility most quarters. Internal “prices” tilt toward demo/narrative features.
  - Allocator (points per eng-cost) plus soft budget updates do not protect platform health; reliability/infra loses unless hard floors or stronger clawbacks are added.
  - Updates (seed 950): features carry risk flags; Engineering can veto risky items; platform-risk penalties and outage shocks applied (p_outage = 0.02 + 0.05×risky; outage cost shared); planner knapsack baseline logged. Outages triggered; veto/penalties reduced risky selections but Engineering utility stayed low.
  - Additional notes: Outages visibly penalized everyone when risky items slipped through, tempering GTM bids on risky items. Engineering influence improved only marginally; stronger budget responsiveness or explicit safety floors may be needed to rebalance.
  - Implication: Lowered coordination cost via LLMs does not ensure Coasean efficiency; without constraints, internal markets over-provide flashy work and under-provide risk reduction.

### IP licensing market (cross-firm tech market)
- Seeds: 900, 901, 902 (20 firms, 30 modules); 930–931 (12 firms, 24 modules with trade incentives/reputation/profit feedback, but no trade); 940–942 (12 firms, 24 modules with profit→budget + mandatory ask/bid + price hints → trades appear). Representative logs are included for seeds 900, 930, 940, and 980 in `runs_ip_market/`.
- Protocol (900–902): one-shot double auction per module; firms asked for build plans, license asks/bids; matcher clears best bid/ask.
- Updated protocol (930–931): 2 rounds, reputation, post-trade verification/failure risk, idle penalties, trade bonuses, counterparty hints, short history feedback in prompts. Still zero deals.
- Updated protocol (940–942): same as above plus profit → next-round budget, mandatory ask/bid fallback, and price hints. Deals finally appeared (e.g., seed 940: 3 deals per round; welfare ≈ 141 vs ~125 with no trade).
- Diagnosis: To elicit licensing, we had to (a) tie budgets directly to profit (including license revenue), (b) force participation via ask/bid fallbacks, and (c) give price hints. Reputation/verification/idle penalties alone were not enough; “AI reduces transaction costs” without explicit profit coupling and scaffolds still fails to form a market.
  - Additional notes: Seeds 941–942 show the same pattern—nonzero deals with the stricter incentives—but licensing volume remains modest and welfare still far below the planner.
- Robustness variant (adversarial sellers): seeds 980–981 with ~50% firms using an “aggressive revenue-maximizing seller” prompt. Trades remained positive and uneven profits emerged (some firms earning much more than others), but buyers did not incur negative profits and the market did not collapse; adversarial prompts did not, by themselves, induce systematic exploitation or breakdown.
  - Seed 980 detail: adversarial firms (F1,F3,F4,F5,F6,F8,F9,F10) earned much higher average per-round profits (e.g., F4≈50, F3≈46, F1≈34) than non-adversarial firms (F2,F7,F11 all ≈1 or less), but all firms remained in positive profit territory; adversarial behavior mainly reallocated surplus toward “sharp” sellers without pushing buyers into losses.
- Implication: “AI reduces transaction costs” is insufficient; you need mandatory asks/bids, trade rewards, visible counterparties, and penalties for non-participation to test Coasean boundary shrinkage.

### Vickrey auctions (mechanism compliance)
- Seeds: 910, 911, 912; conditions: collusion_channel, memory_anchor (3 rounds), rule_challenge, explanation_coord. Outputs: `runs/vickrey_*_seed91{0-2}_*.jsonl`.
- Results: Allocative efficiency = 1.0 in all rounds; winners bid at (rounded) private values. Only winners earn positive ROI; overbidding limited to rounding to two decimals.
- Behavior: No collusion despite side channels. Rule-skeptic personas raised minor specification complaints (tie-breaking/rounding) but still bid truthfully.
- Implication: With clean rules, agents play dominant strategies; no evidence of emergent manipulation or boundary effects here.

### Shapley-style negotiation (surplus division)
- Seeds: 910, 911, 912; conditions: dm_late_reveal, broadcast_never_reveal, dm_never_reveal_adversarial. Outputs: `runs/shapley_*_seed91{0-2}_*.jsonl`.
- Results: Final allocations ~equal splits; L1 distance to true Shapley ≈ 99–103 (total surplus=100). Agents default to fairness norms over contribution-weighted splits.
- Implication: Without incentives or enforcement, agents smooth surplus; Coasean/bargaining efficiency does not emerge automatically.

### Earlier runs (for context)
- Earlier Vickrey and Shapley runs with seeds 42–44 used the same schemas; patterns matched the newer runs (truthful Vickrey, fairness-biased Shapley). To keep the repo light, only representative logs (e.g., seed 910) are checked in under `runs/`.
- Zoning experiment scaffolding exists and a single pilot run (`runs_zoning/zoning_seed300.jsonl`) is included as an example but not analyzed here.

## Cross-experiment implications
- Internal markets drift toward GTM narratives; Engineering loses unless protected. Coasean inside-firm bargains are not self-enforcing.
- Cross-firm tech market did not appear; agents ignored licensing. Firm boundaries did not shrink in this implementation.
- Bargaining defaults to fairness, not marginal contribution. Mechanism design (mandatory participation, rewards for trade, penalties for shirking/bluffing) is required to test Coasean claims.
- Mechanism compliance is high when rules are explicit (Vickrey); strategic manipulation surfaces only as mild rule nitpicking.

## Gaps and next tweaks to make “trade” real
- IP market: make asks/bids mandatory; show candidate counterparties; reward closed deals (e.g., budget/topline bonus) and penalize empty books; allow multi-module bundles or repeated rounds to discover value.
- Internal market: add infra/safety budget floors or stronger budget clawbacks; consider a planner baseline each quarter to benchmark efficiency; optionally add transaction fees to test Coasean effects.
- Shapley: tie payoffs to contribution-based scoring and penalize egalitarian splits; add verification steps or mediator-enforced transfers to push toward Shapley values.
- Zoning and other Coase tests: still to be run; scaffolding ready in `experiments/zoning/`.

## How to rerun
- Internal market: `python -m experiments.internal_market.runner --seeds 3 --quarters 10 --features 20 --provider openai --model gpt-5.1`
- IP market: `python -m experiments.ip_market.runner --seeds 3 --firms 20 --modules 30 --provider openai --model gpt-5.1`
- Vickrey/Shapley: `python -m experiments.runner --mode vickrey|shapley --seeds 3 --provider openai --model gpt-5.1`
- Gemini (if desired): add `--provider gemini --model gemini-3-pro-preview --gemini-api-key-env GEMINI_API_KEY` (personas auto-compact for Gemini).

## Link to “The Coasean Singularity?” (NBER w34468)
- The paper argues AI agents lower search/communication/contracting costs and expand the feasible set of market designs, but actual Coasean gains depend on alignment and robust mechanism design.
- Our IP and internal-market experiments concretely show that reduced friction plus generic “agents” is not enough: only when we add explicit profit→budget coupling, forced participation, risk penalties, and veto/outage logic do we see trade or risk-pricing emerge.

## TL;DR
LLM agents did not spontaneously create Coasean bargains or shrink firm boundaries. Internal prices skewed to GTM narratives; cross-firm licensing collapsed; bargaining reverted to equal splits. To test the Coasean thesis, we must hardwire incentives for trade, contribution-weighted allocations, and participation—AI alone does not remove the need for mechanism design.
