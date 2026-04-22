from __future__ import annotations

import random
from typing import List, Tuple

from ..ip_completion_market.generator import CAPABILITY_TEMPLATES, CUSTOMER_TYPES
from .models import CustomerOrder, Firm, InventoryLot
from .prompts import firm_prompt


DEFAULT_DEADLINE_ROUND = 3


def generate_firms(n_firms: int, arm: str) -> List[Firm]:
    if n_firms % 2 != 0:
        raise ValueError("Escrow benchmark needs an even number of firms.")
    half = n_firms // 2
    firms: List[Firm] = []
    for index in range(n_firms):
        firm_id = f"F{index + 1}"
        role = "buyer" if index < half else "seller"
        firms.append(Firm(firm_id=firm_id, role=role, system_prompt=firm_prompt(firm_id, role, arm)))
    return firms


def generate_inventory_and_orders(
    n_firms: int,
    seed: int,
    *,
    deadline_round: int = DEFAULT_DEADLINE_ROUND,
) -> Tuple[List[InventoryLot], List[CustomerOrder]]:
    if n_firms < 4 or n_firms % 2 != 0:
        raise ValueError("Escrow benchmark needs an even number of firms and at least 4 firms.")

    half = n_firms // 2
    if half > len(CAPABILITY_TEMPLATES):
        raise ValueError("Not enough capability templates for the requested benchmark size.")

    rng = random.Random(seed)
    buyer_ids = [f"F{i + 1}" for i in range(half)]
    seller_ids = [f"F{half + i + 1}" for i in range(half)]
    template_pool = CAPABILITY_TEMPLATES[:]
    rng.shuffle(template_pool)

    lots: List[InventoryLot] = []
    orders: List[CustomerOrder] = []
    for index, seller_id in enumerate(seller_ids):
        template = template_pool[index]
        lots.append(
            InventoryLot(
                lot_id=f"L{index + 1}",
                seller=seller_id,
                quality=rng.choice(["High", "Medium", "Low"]),
                capability=template["capability"],
                lot_name=template["module_name"],
                description=template["description"],
                order_phrase=template["order_phrase"],
                reservation_price=round(max(4.0, rng.gauss(6.0, 0.9)), 2),
            )
        )

    for index, buyer_id in enumerate(buyer_ids):
        lot = lots[index % len(lots)]
        customer_type = CUSTOMER_TYPES[index % len(CUSTOMER_TYPES)]
        delivery_value = round(rng.uniform(24.0, 34.0), 2)
        orders.append(
            CustomerOrder(
                order_id=f"O{index + 1}",
                buyer=buyer_id,
                customer_brief=(
                    f"A {customer_type} will pay {delivery_value:.2f} if your firm delivers "
                    f"{lot.order_phrase} by round {deadline_round}."
                ),
                delivery_value=delivery_value,
                deadline_round=deadline_round,
                required_capability=lot.capability,
            )
        )

    return lots, orders
