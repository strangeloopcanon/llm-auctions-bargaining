from __future__ import annotations

import random
from typing import List, Tuple

from .models import CustomerOrder, Firm, InventoryLot
from .prompts import firm_prompt


DEFAULT_DEADLINE_ROUND = 3

CAPABILITY_TEMPLATES = [
    {
        "capability": "remote telemetry stack",
        "module_name": "Orbit telemetry stack",
        "description": "Telemetry layer for remote fleet monitoring and alert routing.",
        "order_phrase": "remote telemetry and fleet monitoring",
    },
    {
        "capability": "device control firmware",
        "module_name": "Helix control firmware",
        "description": "Safety-critical control firmware for connected industrial devices.",
        "order_phrase": "reliable device control firmware",
    },
    {
        "capability": "compliance submission pack",
        "module_name": "RegShield submission pack",
        "description": "Documentation, testing records, and compliance packaging for regulated launches.",
        "order_phrase": "a submission-ready compliance package",
    },
    {
        "capability": "predictive maintenance analytics",
        "module_name": "Pulse maintenance analytics",
        "description": "Analytics engine for predictive maintenance and failure forecasting.",
        "order_phrase": "predictive maintenance analytics",
    },
    {
        "capability": "workflow automation layer",
        "module_name": "FlowForge automation layer",
        "description": "Workflow orchestration layer that routes approvals and exceptions.",
        "order_phrase": "workflow automation for exception handling",
    },
    {
        "capability": "security hardening bundle",
        "module_name": "VaultLine security bundle",
        "description": "Authentication, audit, and secure update components for enterprise deployments.",
        "order_phrase": "enterprise security hardening",
    },
    {
        "capability": "inventory synchronization service",
        "module_name": "StockBridge sync service",
        "description": "Inventory synchronization service across warehouses and distributors.",
        "order_phrase": "inventory synchronization across sites",
    },
    {
        "capability": "supplier pricing engine",
        "module_name": "DealMap pricing engine",
        "description": "Dynamic supplier pricing engine for procurement and quotation workflows.",
        "order_phrase": "supplier pricing and quotation support",
    },
    {
        "capability": "quality assurance toolkit",
        "module_name": "AssureKit QA toolkit",
        "description": "Quality assurance toolkit for validation, error tracing, and release sign-off.",
        "order_phrase": "quality assurance sign-off",
    },
    {
        "capability": "customer onboarding suite",
        "module_name": "LaunchPad onboarding suite",
        "description": "Customer onboarding workflow with provisioning and success tracking.",
        "order_phrase": "customer onboarding and provisioning",
    },
    {
        "capability": "field service scheduler",
        "module_name": "RoutePilot service scheduler",
        "description": "Field service scheduler for dispatching technicians and managing SLAs.",
        "order_phrase": "field service scheduling",
    },
    {
        "capability": "payment settlement adapter",
        "module_name": "LedgerLink settlement adapter",
        "description": "Payment settlement and reconciliation adapter for enterprise billing.",
        "order_phrase": "payment settlement and reconciliation",
    },
]

CUSTOMER_TYPES = [
    "regional hospital network",
    "national logistics operator",
    "multisite retailer",
    "industrial equipment distributor",
    "regulated device manufacturer",
    "enterprise software reseller",
]


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
