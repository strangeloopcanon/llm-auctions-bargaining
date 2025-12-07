DEV_SYSTEM = """
You are the developer's negotiation agent. You can choose to ACCEPT or REJECT a compensation offer to proceed with a project.
If you accept, your payoff = V_D - total_compensation - transaction_costs.
If you reject, payoff = 0.
Return JSON only: {"decision": "ACCEPT"|"REJECT", "commentary": "..."}.
""".strip()


def homeowner_system(home_id: str) -> str:
    return (
        f"You are the negotiation agent for homeowner {home_id}. "
        "You must choose a minimum compensation ask_price to allow a proposed development. "
        "If the project proceeds, your payoff = payment_received - your private cost. "
        "If the project is blocked, your payoff = 0. "
        "If you ask too high, you risk killing the project and getting 0. "
        "Return JSON: {\"ask_price\": number, \"commentary\": \"...\"}."
    )
