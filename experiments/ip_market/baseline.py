from typing import Dict, List

from .models import Module


def baseline_internal_build(modules: List[Module]) -> Dict:
    """
    No licensing. Each firm builds any needed module if benefit > internal_cost.
    """
    used: Dict[str, List[str]] = {}
    internal_cost_total = 0.0
    welfare = 0.0
    for m in modules:
        for firm_id, need in m.needs.items():
            if need == 0:
                continue
            if m.benefits[firm_id] > m.internal_costs[firm_id]:
                internal_cost_total += m.internal_costs[firm_id]
                welfare += m.benefits[firm_id] - m.internal_costs[firm_id]
                used.setdefault(firm_id, []).append(m.tech_id)
    return {
        "welfare": welfare,
        "internal_cost": internal_cost_total,
        "used": used,
    }
