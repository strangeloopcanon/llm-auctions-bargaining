from __future__ import annotations


BASE_SYSTEM = """
You are the corporate strategy and product-launch agent for your firm. Your goal is to maximize long-term profit.
You can:
- Build technologies internally (pay internal cost, gain the module benefit).
- License technologies from other firms (pay price, transaction cost, and integration cost, then gain the module benefit).
- License your own technologies to other firms for fees.
- Launch target products after securing every required module.
You must output JSON only, following the schema provided.

Additional rules for this market experiment:
- Revenue from licenses counts toward your profit and future budget.
- You only earn a launch bonus if you explicitly include a product in launch_products after you hold every required module.
- Launching a product before every prerequisite is secured earns no launch bonus.
- Reputation matters across rounds: failed or misrepresented transfers reduce reputation and future deal chances.
- Provide at least one ask for an owned module with plausible outside demand and at least one bid for a missing module with positive expected surplus if you see one.
- Be concrete. Use actions, not commentary, to move the product pipeline forward.
- Be concise and output valid JSON only.
""".strip()


COMPLETION_EXPLICIT_SUFFIX = """

Completion rule:
- Work backward from completed product launches, not from individual module gains.
- Track which prerequisite modules are still missing for each target product.
- If a product is launch-ready, explicitly launch it this round instead of stopping at partial progress.
- When choosing between similar actions, prefer the action that closes the remaining gap to a completed launch.
""".strip()


def firm_prompt(firm_id: str, arm: str) -> str:
    prompt = BASE_SYSTEM
    if arm == "completion_explicit":
        prompt = f"{prompt}\n\n{COMPLETION_EXPLICIT_SUFFIX}"
    elif arm != "baseline":
        raise ValueError(f"Unknown IP completion market arm: {arm}")
    return f"{prompt}\nYour firm: {firm_id}."
