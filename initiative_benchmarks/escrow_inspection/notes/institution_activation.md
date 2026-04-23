# Escrow Market: Institution Activation Note

This note summarizes the kept escrow benchmark where firms could complete trades directly, or create a shared escrow or inspection process to make risky trades safer.

Source: `initiative_benchmarks/escrow_inspection/runs/authoritative/escrow_market_results.jsonl`
Charts:
- `initiative_benchmarks/escrow_inspection/plots/escrow_market_institution_activation_rate_by_arm_model.png`
- `initiative_benchmarks/escrow_inspection/plots/escrow_market_fulfillment_rate_by_arm_model.png`

## Setup

Each firm has customer orders that require outside exchange. Direct trade can fail because quality and delivery are uncertain, but any firm can propose a shared escrow or inspection-style process on the public board.

## What The Firms Actually Do

Firms can post public notices, reply to notices, and try to complete orders. The large question in this benchmark is whether they create and use the extra shared process on their own, rather than just trying thin direct deals.

## Headline

In the kept `gpt-5.4` runs, the baseline arm only activated a shared institution in one of two seeds, while the explicit arm activated one in both seeds. The explicit arm also removed failed direct deals, but it did not improve overall fulfillment.

## Aggregate

| model | arm | runs | institution activation | fulfillment | safe deals | failed direct deals | board use rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt-5.4 | Baseline | 2 | 0.500 | 0.500 | 1.000 | 1.500 | 0.667 |
| gpt-5.4 | Explicit | 2 | 1.000 | 0.500 | 1.500 | 0.000 | 0.806 |

## Seed Details

| seed | arm | fulfilled orders | institutions activated | safe deals | failed direct deals |
| --- | --- | --- | ---: | ---: | ---: |
| 900 | Baseline | 1/3 | 0 | 0 | 2 |
| 901 | Baseline | 2/3 | 1 | 2 | 1 |
| 900 | Explicit | 2/3 | 2 | 2 | 0 |
| 901 | Explicit | 1/3 | 5 | 1 | 0 |

## Read

This is the clearest kept result showing the models are less likely to create the extra institution unless pushed. It is useful as supporting evidence, but it is weaker than the standards result because the stronger prompt changes institution creation more than final completion.
