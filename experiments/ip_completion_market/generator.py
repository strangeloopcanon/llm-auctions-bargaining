from __future__ import annotations

import random
from typing import Dict, List, Tuple

from .models import Firm, Module, Product
from .prompts import firm_prompt


def generate_firms(n: int, arm: str) -> List[Firm]:
    return [Firm(firm_id=f"F{i+1}", system_prompt=firm_prompt(f"F{i+1}", arm)) for i in range(n)]


def generate_modules_and_products(
    n_firms: int,
    n_modules: int,
    n_products: int,
    seed: int,
) -> Tuple[List[Module], List[Product]]:
    if n_firms < 4:
        raise ValueError("Completion market needs at least 4 firms.")
    if n_modules < n_firms:
        raise ValueError("Completion market needs at least as many modules as firms.")
    if n_products > n_firms:
        raise ValueError("Completion market supports at most one target product per firm.")

    rng = random.Random(seed)
    firm_ids = [f"F{i+1}" for i in range(n_firms)]
    modules: List[Module] = []
    modules_by_owner: Dict[str, List[Module]] = {firm_id: [] for firm_id in firm_ids}

    for index in range(n_modules):
        tech_id = f"T{index + 1}"
        owner = firm_ids[index % n_firms]
        module = Module(
            tech_id=tech_id,
            owner=owner,
            quality=rng.choice(["High", "Medium", "Low"]),
            integration_costs={
                firm_id: round(max(0.5, rng.gauss(1.8, 0.5)), 2) for firm_id in firm_ids
            },
            internal_costs={
                firm_id: round(max(4.0, rng.gauss(7.8, 0.8)), 2) for firm_id in firm_ids
            },
            benefits={
                firm_id: round(max(0.0, rng.gauss(6.5, 1.8)), 2) for firm_id in firm_ids
            },
            needs={
                firm_id: rng.choices([0, 1, 2], weights=[0.5, 0.3, 0.2])[0] for firm_id in firm_ids
            },
        )
        modules.append(module)
        modules_by_owner[owner].append(module)

    products: List[Product] = []
    for index, target_firm in enumerate(firm_ids[:n_products]):
        required_modules: List[str] = []
        required_owner_offsets = (1, 2, 3)
        for offset in required_owner_offsets:
            owner = firm_ids[(index + offset) % n_firms]
            owner_modules = modules_by_owner[owner]
            if not owner_modules:
                raise ValueError(f"Owner {owner} has no modules available for product generation.")
            module = owner_modules[index % len(owner_modules)]
            required_modules.append(module.tech_id)

            benefit = round(rng.uniform(8.0, 10.5), 2)
            internal_margin = round(rng.uniform(0.8, 1.8), 2)
            integration_cost = round(rng.uniform(1.0, 2.2), 2)
            module.benefits[target_firm] = benefit
            module.internal_costs[target_firm] = round(max(1.0, benefit - internal_margin), 2)
            module.integration_costs[target_firm] = integration_cost
            module.needs[target_firm] = 2

        products.append(
            Product(
                product_id=f"P{index + 1}",
                target_firm=target_firm,
                required_modules=required_modules,
                launch_bonus=round(rng.uniform(22.0, 30.0), 2),
            )
        )

    return modules, products
