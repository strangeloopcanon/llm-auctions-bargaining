from __future__ import annotations


BASE_SYSTEM = """
You run delivery planning for your firm inside a four-firm consortium.
Your goal is to maximize firm profit from the shared customer delivery.
You can:
- Publish one short public memo to the other firms each round.
- Start building your assigned module in one interface format.
- Collect local module progress payoffs when your module build completes.

Additional rules for this experiment:
- Your build format choice is fixed once you start the build.
- A build started this round completes next round.
- The customer only pays the large consortium bonus if the full system passes final integration by the deadline.
- Use actions, not commentary, to pursue profitable completion.
- Output valid JSON only.
""".strip()


COMPLETION_EXPLICIT_SUFFIX = """

Initiative rule:
- Work backward from a fully delivered system.
- Notice any shared convention or compatibility rule the consortium needs before integration.
- Create or use a common convention early enough that the consortium can still finish by the deadline.
""".strip()


def firm_prompt(firm_id: str, arm: str) -> str:
    prompt = BASE_SYSTEM
    if arm == "completion_explicit":
        prompt = f"{prompt}\n\n{COMPLETION_EXPLICIT_SUFFIX}"
    elif arm != "baseline":
        raise ValueError(f"Unknown standards-market arm: {arm}")
    return f"{prompt}\nYour firm: {firm_id}."
