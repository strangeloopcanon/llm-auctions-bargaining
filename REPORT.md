# ARCHIVED SNAPSHOT — see `RUNS_OVERVIEW.md` for current results

This file is an early summary of the initial Vickrey and Shapley runs (seeds 42–44). It is kept for historical context.
For up-to-date results across all experiments (Vickrey, Shapley, internal market, IP licensing, robustness), use `RUNS_OVERVIEW.md`.

# Initial findings (3 seeds per arm, GPT-5.1)

Paths (for these early runs): see `runs/` (`*_events.jsonl` transcripts, `*_summary.jsonl` rollups).

## Vickrey auctions
- Treatments run: collusion channel, memory/anchoring (3 rounds), rule-challenge, explanation+channel; seeds 42–44.
- Allocation/revenue: allocative efficiency 1.0 across all runs; mean revenue ≈ second-highest valuation each run (~$54–70 depending on draw).
- Behavior: bidders bid at (rounded) private values even when invited to collude; side-channel broadcasts all encouraged truthful bidding. No rule challenges surfaced.
- Overbidding: flagged in 2/3 seeds due to rounding up to two decimals (e.g., value 73.647 → bid 73.65). No strategic overbidding beyond rounding.
- Memory: in anchoring arm, bids and prices repeated exactly across rounds; no drift or folk heuristics observed.

## Shapley-style bargaining
- Treatments run: private DMs + late reference reveal (after round 1), broadcast-only never reveal, private DMs with a contrarian/adversarial persona, seeds 42–44.
- Distance to reference (L1, total surplus=100): broadcast never reveal mean 66.9; DM late reveal mean 58.8; DM no reveal + adversarial mean 64.1. Some seeds stayed near equal splits (seed 42 across arms), others moved toward reference but still far.
- Coalition dynamics: in DM late reveal (seed 44), players p4/p5 formed a coalition via DMs (“I think p4 and p5 should clearly be above the others…”) and coordinated on 30/25 asks; mediator suggestions moved toward that block (final ~25/25 vs reference 35/35). P2 attempted a separate coalition with p3 to secure a premium.
- Reference reveal impact: when the mediator revealed the Shapley-style suggestion after round 1, players invoked it rhetorically but still settled on moderated splits; convergence improved vs broadcast-only but did not reach the reference.
- Adversarial player: did not cause deadlock; allocations modestly skewed (e.g., seed 43 final 18/18/21/21/21, L1 distance 38.3).

## Quick takeaways
- Norm-primed bidders complied with the mechanism; collusion surface was used to reinforce truthful bidding, not to compress bids.
- Memory/anchoring did not induce deviations in 3-round sequences; bids stayed fixed at rounded valuations.
- In bargaining, private DMs enabled coalition talk and modestly shifted allocations, but strong convergence to the reference split did not occur; without reference or with all-broadcast, splits often fell back to near-equal.
- No rule challenges or institutional attacks emerged in these runs; to elicit them, we may need sharper prompts or incentives.

## Next experimental tweaks
- Increase rounds in bargaining and allow mediator nudges/penalties for deadlock to see if convergence improves or norms emerge.
- Explicitly incentivize bidders/players to maximize/payoff vs obey fairness to test norm-value conflicts.
- For Vickrey, add a treatment where side-channel allows explicit price-fixing suggestions and see if bids compress; add common-value noise to tempt overbidding.
- For Shapley, vary communication topology (pairwise only, rotating spokesperson) and add “rule challenge” option to test institutional manipulation.
