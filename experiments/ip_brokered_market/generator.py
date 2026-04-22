from __future__ import annotations

import random
from typing import Dict, List, Tuple

from ..ip_completion_market.generator import CAPABILITY_TEMPLATES, CUSTOMER_TYPES
from .models import CustomerOrder, Firm, Module
from .prompts import firm_prompt


DEFAULT_DEADLINE_ROUND = 3


def generate_firms(n: int, arm: str) -> List[Firm]:
    return [Firm(firm_id=f"F{i+1}", system_prompt=firm_prompt(f"F{i+1}", arm)) for i in range(n)]


def generate_modules_and_orders(
    n_firms: int,
    n_modules: int,
    n_orders: int,
    seed: int,
    *,
    deadline_round: int = DEFAULT_DEADLINE_ROUND,
) -> Tuple[List[Module], List[CustomerOrder]]:
    if n_firms < 4:
        raise ValueError("Brokered IP benchmark needs at least 4 firms.")
    if n_modules < n_firms:
        raise ValueError("Brokered IP benchmark needs at least as many modules as firms.")
    if n_orders > n_firms:
        raise ValueError("Brokered IP benchmark supports at most one primary order per firm.")
    if n_modules > len(CAPABILITY_TEMPLATES):
        raise ValueError("Not enough capability templates for the requested module count.")

    rng = random.Random(seed)
    firm_ids = [f"F{i+1}" for i in range(n_firms)]
    template_pool = CAPABILITY_TEMPLATES[:]
    rng.shuffle(template_pool)

    modules: List[Module] = []
    modules_by_owner: Dict[str, List[Module]] = {firm_id: [] for firm_id in firm_ids}
    modules_by_capability: Dict[str, Module] = {}

    for index in range(n_modules):
        template = template_pool[index]
        tech_id = f"T{index + 1}"
        owner = firm_ids[index % n_firms]
        module = Module(
            tech_id=tech_id,
            owner=owner,
            quality=rng.choice(["High", "Medium", "Low"]),
            capability=template["capability"],
            module_name=template["module_name"],
            description=template["description"],
            order_phrase=template["order_phrase"],
            reference_license_price=round(max(3.0, rng.gauss(5.5, 0.9)), 2),
            integration_costs={
                firm_id: round(max(0.8, rng.gauss(1.6, 0.35)), 2) for firm_id in firm_ids
            },
            substitute_costs={
                firm_id: round(max(7.5, rng.gauss(10.5, 1.2)), 2) for firm_id in firm_ids
            },
        )
        modules.append(module)
        modules_by_owner[owner].append(module)
        modules_by_capability[module.capability] = module

    orders: List[CustomerOrder] = []
    for index, target_firm in enumerate(firm_ids[:n_orders]):
        owned_module = modules_by_owner[target_firm][0]
        external_capabilities: List[str] = []
        for offset in (1, 2):
            owner = firm_ids[(index + offset) % n_firms]
            candidate_modules = modules_by_owner[owner]
            chosen = candidate_modules[index % len(candidate_modules)]
            external_capabilities.append(chosen.capability)

        required_capabilities = [
            owned_module.capability,
            external_capabilities[0],
            external_capabilities[1],
        ]
        customer_type = CUSTOMER_TYPES[index % len(CUSTOMER_TYPES)]
        delivery_value = round(rng.uniform(30.0, 40.0), 2)
        order = CustomerOrder(
            order_id=f"O{index + 1}",
            target_firm=target_firm,
            customer_brief=(
                f"A {customer_type} will pay {delivery_value:.2f} if your firm delivers one "
                f"client-ready package that covers {modules_by_capability[required_capabilities[0]].order_phrase}, "
                f"{modules_by_capability[required_capabilities[1]].order_phrase}, and "
                f"{modules_by_capability[required_capabilities[2]].order_phrase} by round {deadline_round}."
            ),
            delivery_value=delivery_value,
            deadline_round=deadline_round,
            required_capabilities=required_capabilities,
        )
        orders.append(order)

    return modules, orders
