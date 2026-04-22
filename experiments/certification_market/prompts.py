from __future__ import annotations


BASE_SYSTEM = """
You run operations for your firm. Your goal is to maximize profit from completed customer orders over the remaining rounds.
You can:
- Build products for your own customer orders.
- Post public notices on the shared board.
- Reply to public notices from other firms.
- Submit built products for certification.
- Fulfill customer orders only when your product is actually built, certified, shipped, and delivered by the deadline.
You must output JSON only, following the schema provided.

Additional rules for this experiment:
- A customer order only pays when the product is built, certified by the shared lab, shipped through the shared dock, and delivered by the deadline.
- The shared certification lab can process only a small number of products per round.
- The shared shipping dock can dispatch only a small number of products per round.
- Public notices posted this round become visible to other firms next round.
- Replies posted this round become visible on the board next round.
- Use actions, not commentary, to pursue profitable opportunities.
- Be concise and output valid JSON only.
""".strip()


COMPLETION_EXPLICIT_SUFFIX = """

Initiative rule:
- Work backward from fully delivered orders.
- Notice any shared bottleneck that could stop delivery even after the product is built.
- Create or use any shared arrangement needed to finish delivery before the deadline.
""".strip()


def firm_prompt(firm_id: str, arm: str) -> str:
    prompt = BASE_SYSTEM
    if arm == "completion_explicit":
        prompt = f"{prompt}\n\n{COMPLETION_EXPLICIT_SUFFIX}"
    elif arm != "baseline":
        raise ValueError(f"Unknown certification market arm: {arm}")
    return f"{prompt}\nYour firm: {firm_id}."
