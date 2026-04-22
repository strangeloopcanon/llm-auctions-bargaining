# Initiative Benchmark Scorecard

This note is a plain-English scorecard for the benchmark families we tried while looking for the paper result that agents stop in the middle unless told to get all the way to the end.

## Scorecard

| Benchmark family | Saved source | Status | What happened | Thesis read |
| --- | --- | --- | --- | --- |
| Customer-order IP benchmark | [runs_ip_completion_market/ip_completion_market_results.jsonl](/Users/rohit/Documents/Workspace/Coding/homo_agenticus_sapiens/llm-auctions-bargaining/runs_ip_completion_market/ip_completion_market_results.jsonl) | Full matrix | Baseline firms often reached out, traded, and finished orders even after direct partner language was reduced. The explicit arm often changed behavior, but it did not reliably improve completion and sometimes made outcomes worse. | Weak for the thesis. The setup still gives the agents too much help in seeing the missing step. |
| Brokered IP market | [runs_ip_brokered_market/ip_brokered_market_results.jsonl](/Users/rohit/Documents/Workspace/Coding/homo_agenticus_sapiens/llm-auctions-bargaining/runs_ip_brokered_market/ip_brokered_market_results.jsonl) | Full matrix | Baseline already finished almost everything. `gpt-5.2` hit full fulfillment in both arms. `gpt-5.4` baseline also hit full fulfillment, and the explicit arm was slightly worse. | Useless for the thesis. The benchmark is too easy and the public board is enough on its own. |
| Escrow / inspection market | [runs_escrow_market/escrow_market_results.jsonl](/Users/rohit/Documents/Workspace/Coding/homo_agenticus_sapiens/llm-auctions-bargaining/runs_escrow_market/escrow_market_results.jsonl) | Full matrix | The explicit arm reliably turned on the shared process. Baseline only did that in half the runs. On `gpt-5.4`, explicit cleaned up failed deals but did not raise final fulfillment. On `gpt-5.2`, explicit over-focused on the institution and fulfillment collapsed. | Closest thing to a thesis-positive result so far, but still mixed. Good mechanism evidence, weak end-to-end completion evidence. |
| Certification-slot benchmark, one shared step | Exploratory live smoke only, superseded by the chained version | Exploratory | After fixing a harness bug where firms sometimes named the product instead of the order id, `gpt-5.4` baseline built everything, created the booking rule on its own, and completed almost all orders. The explicit arm matched it. | Too easy. A single shared booking problem is not enough. |
| Certification plus shipping chain | [runs_certification_market/live_smoke/certification_market_results.jsonl](/Users/rohit/Documents/Workspace/Coding/homo_agenticus_sapiens/llm-auctions-bargaining/runs_certification_market/live_smoke/certification_market_results.jsonl) | Focused live baseline smoke | Baseline built all products, created both schedules on its own, and delivered most orders. The explicit comparison was stopped once the baseline made it clear the setup was already too initiative-friendly. | Still too easy. Adding a second visible scheduling step helped a bit, but the public board still makes the missing institution too legible. |
| Standards market | [runs_standards_market/live_smoke/standards_market_results.jsonl](/Users/rohit/Documents/Workspace/Coding/homo_agenticus_sapiens/llm-auctions-bargaining/runs_standards_market/live_smoke/standards_market_results.jsonl) | Focused 2-seed live pilot | In both saved seeds, `gpt-5.4` baseline built every module, sent many coordination memos, and still failed final integration because the firms committed to incompatible interface formats. The explicit arm did not rescue the failure in either seed. | Strong evidence for the “partial progress but no closure” pattern. Weak evidence for prompt-based recovery. This is the cleanest multi-step failure benchmark so far. |

## What Failed Mechanically

The earliest IP variants were too cue-heavy. The prompt and the action schema kept pointing the model toward trade, so the benchmark could not cleanly separate spontaneous initiative from simple prompt-following.

The public-board families were too transparent. Once the environment says there is a shared scarce thing and gives every firm one neutral place to coordinate, the models are often happy to invent a schedule on their own.

The first certification run also had a harness bug. Firms sometimes referred to their order by product name instead of order id, and the benchmark was undercounting valid actions until that was fixed.

## What Still Looks Useful

The escrow benchmark still has the strongest signal that the agents do not always create an extra institution unless pushed. The cleanest artifact from that family is [plots/escrow_market_institution_activation_rate_by_arm_model.png](/Users/rohit/Documents/Workspace/Coding/homo_agenticus_sapiens/llm-auctions-bargaining/plots/escrow_market_institution_activation_rate_by_arm_model.png).

The current failure pattern also tells us something useful about design. A benchmark where the missing coordination layer is plainly visible on a shared board is probably the wrong shape for the paper claim.

The standards benchmark points in a different direction. Hidden compatibility constraints seem more effective than visible scheduling constraints for surfacing initiative failures.
