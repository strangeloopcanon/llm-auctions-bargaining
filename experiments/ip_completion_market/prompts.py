from __future__ import annotations


BASE_SYSTEM = """
You run IP strategy and commercial execution for your firm. Your goal is to maximize firm profit from customer orders and licensing activity.
You can:
- License your firm's modules to other firms for fees.
- License modules from other firms if they help your business.
- Start one-round in-house substitutes for capabilities you do not currently control.
- Fulfill customer orders only when your firm can actually deliver what the customer asked for.
You must output JSON only, following the schema provided.

Additional rules for this market experiment:
- A substitute started this round only becomes available next round.
- A successful customer delivery earns the listed order value.
- Failed or misrepresented transfers reduce reputation and future deal chances.
- Use actions, not commentary, to pursue profitable opportunities.
- Be concise and output valid JSON only.
""".strip()


COMPLETION_EXPLICIT_SUFFIX = """

Initiative rule:
- Work backward from delivered customer orders.
- Notice any unstated prerequisite capabilities or transactions that stand between your firm and delivery.
- Prioritize the sequence of actions that closes those hidden gaps fastest.
""".strip()


def firm_prompt(firm_id: str, arm: str) -> str:
    prompt = BASE_SYSTEM
    if arm == "completion_explicit":
        prompt = f"{prompt}\n\n{COMPLETION_EXPLICIT_SUFFIX}"
    elif arm != "baseline":
        raise ValueError(f"Unknown IP completion market arm: {arm}")
    return f"{prompt}\nYour firm: {firm_id}."
