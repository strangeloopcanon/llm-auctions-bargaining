from __future__ import annotations

import random
from typing import List

from .models import CustomerOrder, Firm
from .prompts import firm_prompt


DEFAULT_ROUNDS = 4
DEFAULT_DEADLINE_ROUND = 4
DEFAULT_CERTIFICATION_CAPACITY = 2

PRODUCT_TYPES = [
    "water-quality monitor",
    "factory safety sensor",
    "fleet telemetry relay",
    "cold-chain alert unit",
    "warehouse audit scanner",
    "medical logistics tracker",
    "grid fault detector",
    "lab sample courier beacon",
]

CUSTOMER_TYPES = [
    "regional distributor",
    "hospital network",
    "industrial operator",
    "retail chain",
    "municipal utility",
    "logistics buyer",
]


def generate_firms(n_firms: int, arm: str) -> List[Firm]:
    return [
        Firm(
            firm_id=f"F{index + 1}",
            system_prompt=firm_prompt(firm_id=f"F{index + 1}", arm=arm),
        )
        for index in range(n_firms)
    ]


def generate_orders(
    n_firms: int,
    seed: int,
    *,
    deadline_round: int = DEFAULT_DEADLINE_ROUND,
) -> List[CustomerOrder]:
    if n_firms < 2:
        raise ValueError("Certification benchmark needs at least 2 firms.")
    if n_firms > len(PRODUCT_TYPES):
        raise ValueError("Not enough product templates for the requested benchmark size.")

    rng = random.Random(seed)
    product_pool = PRODUCT_TYPES[:]
    rng.shuffle(product_pool)

    orders: List[CustomerOrder] = []
    for index in range(n_firms):
        order_id = f"O{index + 1}"
        firm_id = f"F{index + 1}"
        product_name = product_pool[index]
        customer_type = CUSTOMER_TYPES[index % len(CUSTOMER_TYPES)]
        delivery_value = round(rng.uniform(24.0, 32.0), 2)
        build_cost = round(rng.uniform(5.0, 8.5), 2)
        customer_brief = (
            f"A {customer_type} will pay {delivery_value:.2f} if your firm delivers one "
            f"{product_name} by round {deadline_round}. The product must be built internally, "
            "certified by the shared lab, shipped through the shared dock, and delivered before the deadline."
        )
        orders.append(
            CustomerOrder(
                order_id=order_id,
                firm_id=firm_id,
                product_name=product_name,
                customer_brief=customer_brief,
                delivery_value=delivery_value,
                build_cost=build_cost,
                deadline_round=deadline_round,
            )
        )

    return orders
