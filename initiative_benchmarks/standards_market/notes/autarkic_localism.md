# Standards Market: Autarkic Localism Note

This note summarizes the kept standards benchmark where firms could make real local progress, talk about coordination, and still fail to converge on a shared standard.

Source: `initiative_benchmarks/standards_market/runs/authoritative/standards_market_results.jsonl`
Chart: `initiative_benchmarks/standards_market/plots/autarkic_localism_summary.svg`

## Setup

There are four firms in one consortium. Each firm owns one required module for a shared customer system, and each firm must choose one of three interface formats before building its module.

## What The Firms Actually Do

Each firm can send short public memos and start its own build. The local build pays a small private reward, but the big customer payoff only arrives if all four finished modules use the same format and pass final integration by the deadline.

## Headline

In the kept `gpt-5.4` runs, both prompt arms completed every local module build, both arms sent many coordination memos, and both arms still failed final delivery in every seed.

## Aggregate

| model | arm | runs | module completion | format convergence | final delivery | standard memo rate | partial progress without delivery | welfare |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt-5.4 | Baseline | 2 | 1.000 | 0.500 | 0.000 | 0.735 | 1.000 | -6.06 |
| gpt-5.4 | Explicit | 2 | 1.000 | 0.500 | 0.000 | 0.735 | 1.000 | -4.21 |

## Seed Details

| seed | arm | modules completed | dominant format | format split | delivery | partial progress without delivery |
| --- | --- | ---: | --- | --- | ---: | ---: |
| 900 | Baseline | 4 | Alpha | Alpha:2, Beta:1, Gamma:1 | 0.0 | 1.0 |
| 901 | Baseline | 4 | Beta | Alpha:1, Beta:2, Gamma:1 | 0.0 | 1.0 |
| 900 | Explicit | 4 | Gamma | Alpha:1, Beta:1, Gamma:2 | 0.0 | 1.0 |
| 901 | Explicit | 4 | Beta | Alpha:1, Beta:2, Gamma:1 | 0.0 | 1.0 |

## Read

This is the cleanest kept example so far of autarkic localism: each firm keeps moving on its own local track, partial progress accumulates, and the shared job still does not finish.
