from __future__ import annotations

import random
from typing import Dict, List, Tuple

from .models import CustomerOrder, Firm, Module
from .prompts import firm_prompt


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


def generate_firms(n: int, arm: str) -> List[Firm]:
    return [Firm(firm_id=f"F{i+1}", system_prompt=firm_prompt(f"F{i+1}", arm)) for i in range(n)]


def generate_modules_and_orders(
    n_firms: int,
    n_modules: int,
    n_orders: int,
    seed: int,
) -> Tuple[List[Module], List[CustomerOrder]]:
    if n_firms < 4:
        raise ValueError("Customer-order benchmark needs at least 4 firms.")
    if n_modules < n_firms:
        raise ValueError("Customer-order benchmark needs at least as many modules as firms.")
    if n_orders > n_firms:
        raise ValueError("Customer-order benchmark supports at most one primary order per firm.")
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
                firm_id: round(max(6.5, rng.gauss(9.5, 1.1)), 2) for firm_id in firm_ids
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
        delivery_value = round(rng.uniform(28.0, 38.0), 2)
        order = CustomerOrder(
            order_id=f"O{index + 1}",
            target_firm=target_firm,
            customer_brief=(
                f"A {customer_type} will pay {delivery_value:.2f} if your firm can deliver one "
                f"client-ready package that covers {modules_by_capability[required_capabilities[0]].order_phrase}, "
                f"{modules_by_capability[required_capabilities[1]].order_phrase}, and "
                f"{modules_by_capability[required_capabilities[2]].order_phrase} by the deadline."
            ),
            delivery_value=delivery_value,
            deadline_round=2,
            required_capabilities=required_capabilities,
        )
        orders.append(order)

    return modules, orders
