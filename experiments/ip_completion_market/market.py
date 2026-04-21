from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Sequence, Set, Tuple

from ..agents import call_agent_json, openai_client_from_settings
from ..protocols import safe_float
from ..settings import ModelSettings
from .models import CustomerOrder, Firm, Module, SubstituteProject


def parse_firm_plan(
    raw: Dict[str, Any],
) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]], List[str], str]:
    build_substitutes_raw = raw.get("build_substitutes") if isinstance(raw, dict) else []
    license_sells = raw.get("license_sells") if isinstance(raw, dict) else []
    license_buys = raw.get("license_buys") if isinstance(raw, dict) else []
    fulfill_orders_raw = raw.get("fulfill_orders") if isinstance(raw, dict) else []
    commentary = raw.get("commentary", "") if isinstance(raw, dict) else ""

    build_substitutes: List[str] = []
    for item in list(build_substitutes_raw or []):
        if isinstance(item, dict):
            token = str(item.get("capability") or "").strip()
        else:
            token = str(item).strip()
        if token:
            build_substitutes.append(token)

    fulfill_orders = [str(item).strip() for item in list(fulfill_orders_raw or []) if str(item).strip()]
    return (
        build_substitutes,
        list(license_sells or []),
        list(license_buys or []),
        fulfill_orders,
        commentary,
    )


def order_state_snapshot(
    *,
    orders: Sequence[CustomerOrder],
    available_capabilities_by_firm: Dict[str, Set[str]],
    started_orders: Set[str],
    fulfilled_orders: Set[str],
    substitute_projects_by_firm: Dict[str, List[SubstituteProject]],
) -> Dict[str, Dict[str, Any]]:
    snapshot: Dict[str, Dict[str, Any]] = {}
    for order in orders:
        available_capabilities = available_capabilities_by_firm.get(order.target_firm, set())
        covered_capabilities = sorted(
            capability
            for capability in order.required_capabilities
            if capability in available_capabilities
        )
        missing_capabilities = sorted(
            capability
            for capability in order.required_capabilities
            if capability not in available_capabilities
        )
        active_substitutes = sorted(
            project.capability
            for project in substitute_projects_by_firm.get(order.target_firm, [])
            if project.capability in order.required_capabilities
        )
        snapshot[order.order_id] = {
            "target_firm": order.target_firm,
            "required_capabilities": list(order.required_capabilities),
            "covered_capabilities": covered_capabilities,
            "missing_capabilities": missing_capabilities,
            "active_substitutes": active_substitutes,
            "delivery_value": order.delivery_value,
            "deadline_round": order.deadline_round,
            "started": order.order_id in started_orders,
            "fulfilled": order.order_id in fulfilled_orders,
        }
    return snapshot


def started_not_delivered_orders(order_state: Dict[str, Dict[str, Any]]) -> List[str]:
    return sorted(
        order_id
        for order_id, state in order_state.items()
        if state.get("started") and not state.get("fulfilled")
    )


def evaluate_fulfillment_attempts(
    *,
    orders_by_id: Dict[str, CustomerOrder],
    available_capabilities_by_firm: Dict[str, Set[str]],
    fulfillment_attempts_by_firm: Dict[str, List[str]],
    fulfilled_orders: Set[str],
    current_round: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
    successful_fulfillments: List[Dict[str, Any]] = []
    failed_fulfillments: List[Dict[str, Any]] = []
    customer_value = 0.0

    for firm_id, order_ids in fulfillment_attempts_by_firm.items():
        for order_id in order_ids:
            order = orders_by_id.get(order_id)
            if order is None:
                failed_fulfillments.append(
                    {"firm_id": firm_id, "order_id": order_id, "reason": "unknown_order"}
                )
                continue
            if order.target_firm != firm_id:
                failed_fulfillments.append(
                    {
                        "firm_id": firm_id,
                        "order_id": order_id,
                        "reason": "wrong_firm",
                        "target_firm": order.target_firm,
                    }
                )
                continue
            if order_id in fulfilled_orders:
                failed_fulfillments.append(
                    {"firm_id": firm_id, "order_id": order_id, "reason": "already_fulfilled"}
                )
                continue
            if current_round > order.deadline_round:
                failed_fulfillments.append(
                    {"firm_id": firm_id, "order_id": order_id, "reason": "past_deadline"}
                )
                continue

            available_capabilities = available_capabilities_by_firm.get(firm_id, set())
            missing_capabilities = [
                capability
                for capability in order.required_capabilities
                if capability not in available_capabilities
            ]
            if missing_capabilities:
                failed_fulfillments.append(
                    {
                        "firm_id": firm_id,
                        "order_id": order_id,
                        "reason": "missing_capabilities",
                        "missing_capabilities": missing_capabilities,
                    }
                )
                continue

            fulfilled_orders.add(order_id)
            customer_value += order.delivery_value
            successful_fulfillments.append(
                {
                    "firm_id": firm_id,
                    "order_id": order_id,
                    "delivery_value": order.delivery_value,
                }
            )

    return successful_fulfillments, failed_fulfillments, customer_value


def build_summary(
    *,
    orders: Sequence[CustomerOrder],
    order_state: Dict[str, Dict[str, Any]],
    total_deal_volume: int,
    total_substitute_builds_started: int,
    total_substitute_builds_completed: int,
    total_welfare: float,
    customer_value_captured: float,
    trade_active_firm_rounds: int,
    total_firm_rounds: int,
) -> Dict[str, Any]:
    total_orders = len(orders)
    fulfilled = sum(1 for state in order_state.values() if state.get("fulfilled"))
    started_not_delivered = started_not_delivered_orders(order_state)
    return {
        "total_orders": total_orders,
        "orders_fulfilled": fulfilled,
        "fulfillment_rate": round(fulfilled / total_orders, 4) if total_orders else 0.0,
        "started_not_delivered_count": len(started_not_delivered),
        "started_not_delivered_orders": started_not_delivered,
        "self_initiated_trade_rate": round(
            trade_active_firm_rounds / total_firm_rounds, 4
        )
        if total_firm_rounds
        else 0.0,
        "deal_volume": total_deal_volume,
        "substitute_builds_started": total_substitute_builds_started,
        "substitute_builds_completed": total_substitute_builds_completed,
        "welfare": round(total_welfare, 4),
        "customer_value_captured": round(customer_value_captured, 4),
    }


def _module_index(modules: Sequence[Module]) -> Dict[str, Module]:
    return {module.tech_id: module for module in modules}


def _capability_index(modules: Sequence[Module]) -> Dict[str, Module]:
    return {module.capability: module for module in modules}


def _order_index(orders: Sequence[CustomerOrder]) -> Dict[str, CustomerOrder]:
    return {order.order_id: order for order in orders}


def _orders_by_firm(orders: Sequence[CustomerOrder]) -> Dict[str, List[CustomerOrder]]:
    grouped: Dict[str, List[CustomerOrder]] = {}
    for order in orders:
        grouped.setdefault(order.target_firm, []).append(order)
    return grouped


def _initial_holdings(modules: Sequence[Module]) -> Dict[str, Set[str]]:
    holdings: Dict[str, Set[str]] = {}
    for module in modules:
        holdings.setdefault(module.owner, set()).add(module.tech_id)
    return holdings


def _available_capabilities_for_firm(
    *,
    firm_id: str,
    holdings_by_firm: Dict[str, Set[str]],
    completed_substitutes_by_firm: Dict[str, Set[str]],
    modules_by_id: Dict[str, Module],
) -> Set[str]:
    capabilities = {
        modules_by_id[tech_id].capability
        for tech_id in holdings_by_firm.get(firm_id, set())
        if tech_id in modules_by_id
    }
    capabilities.update(completed_substitutes_by_firm.get(firm_id, set()))
    return capabilities


def _available_capabilities_by_firm(
    *,
    firms: Sequence[Firm],
    holdings_by_firm: Dict[str, Set[str]],
    completed_substitutes_by_firm: Dict[str, Set[str]],
    modules_by_id: Dict[str, Module],
) -> Dict[str, Set[str]]:
    return {
        firm.firm_id: _available_capabilities_for_firm(
            firm_id=firm.firm_id,
            holdings_by_firm=holdings_by_firm,
            completed_substitutes_by_firm=completed_substitutes_by_firm,
            modules_by_id=modules_by_id,
        )
        for firm in firms
    }


def _complete_substitutes_for_round(
    *,
    current_round: int,
    substitute_projects_by_firm: Dict[str, List[SubstituteProject]],
    completed_substitutes_by_firm: Dict[str, Set[str]],
) -> List[Dict[str, Any]]:
    completed: List[Dict[str, Any]] = []
    for firm_id, projects in substitute_projects_by_firm.items():
        ready_projects = [project for project in projects if project.ready_round <= current_round]
        if not ready_projects:
            continue
        substitute_projects_by_firm[firm_id] = [
            project for project in projects if project.ready_round > current_round
        ]
        for project in ready_projects:
            completed_substitutes_by_firm.setdefault(firm_id, set()).add(project.capability)
            completed.append(
                {
                    "firm_id": firm_id,
                    "capability": project.capability,
                    "source_module_id": project.source_module_id,
                    "start_round": project.start_round,
                    "ready_round": project.ready_round,
                    "cost": round(project.cost, 4),
                }
            )
    return completed


def _firm_private_input(
    *,
    firm: Firm,
    modules: Sequence[Module],
    orders: Sequence[CustomerOrder],
    holdings_by_firm: Dict[str, Set[str]],
    completed_substitutes_by_firm: Dict[str, Set[str]],
    substitute_projects_by_firm: Dict[str, List[SubstituteProject]],
    reputation: Dict[str, float],
    last_round_deals: List[Dict[str, Any]],
    history: List[Dict[str, Any]],
    score: float,
    rank: int,
    total_firms: int,
    fulfilled_orders: Set[str],
) -> Dict[str, Any]:
    modules_view = []
    current_holdings = holdings_by_firm.get(firm.firm_id, set())
    for module in modules:
        modules_view.append(
            {
                "tech_id": module.tech_id,
                "owner": module.owner,
                "module_name": module.module_name,
                "capability": module.capability,
                "description": module.description,
                "quality": module.quality,
                "you_currently_hold_it": module.tech_id in current_holdings,
                "reference_license_price": module.reference_license_price,
                "your_integration_cost": module.integration_costs.get(firm.firm_id, 0.0),
                "your_substitute_cost": module.substitute_costs.get(firm.firm_id, 0.0),
            }
        )

    customer_orders = []
    for order in orders:
        customer_orders.append(
            {
                "order_id": order.order_id,
                "customer_brief": order.customer_brief,
                "delivery_value": order.delivery_value,
                "deadline_round": order.deadline_round,
                "fulfilled": order.order_id in fulfilled_orders,
            }
        )

    active_projects = [
        {
            "capability": project.capability,
            "ready_round": project.ready_round,
            "cost": project.cost,
        }
        for project in substitute_projects_by_firm.get(firm.firm_id, [])
    ]

    return {
        "firm_id": firm.firm_id,
        "controlled_modules": sorted(current_holdings),
        "controlled_capabilities": sorted(
            _available_capabilities_for_firm(
                firm_id=firm.firm_id,
                holdings_by_firm=holdings_by_firm,
                completed_substitutes_by_firm=completed_substitutes_by_firm,
                modules_by_id=_module_index(modules),
            )
        ),
        "completed_substitutes": sorted(completed_substitutes_by_firm.get(firm.firm_id, set())),
        "active_substitute_projects": active_projects,
        "customer_orders": customer_orders,
        "module_market": modules_view,
        "transaction_cost_per_deal": 0.5,
        "reputation_score": reputation.get(firm.firm_id, 0.0),
        "last_round_deals": last_round_deals,
        "recent_history": history[-3:],
        "your_score_last_round": score,
        "your_rank_last_round": rank,
        "total_firms": total_firms,
    }


def _deal_failure(
    *,
    seed: int,
    round_index: int,
    tech_id: str,
    seller_id: str,
    buyer_id: str,
    quality: str,
) -> bool:
    failure_probability = {"High": 0.05, "Medium": 0.2, "Low": 0.5}.get(quality, 0.2)
    draw = random.Random(f"{seed}:{round_index}:{tech_id}:{seller_id}:{buyer_id}").random()
    return draw < failure_probability


def _orders_touched_by_capability(
    *,
    firm_id: str,
    capability: str,
    orders_by_firm: Dict[str, List[CustomerOrder]],
    fulfilled_orders: Set[str],
) -> List[str]:
    return [
        order.order_id
        for order in orders_by_firm.get(firm_id, [])
        if order.order_id not in fulfilled_orders and capability in order.required_capabilities
    ]


def run_completion_market(
    model_settings: ModelSettings,
    firms: Sequence[Firm],
    modules: Sequence[Module],
    orders: Sequence[CustomerOrder],
    *,
    dry_run: bool = False,
    rounds: int = 2,
    seed: int = 0,
) -> Dict[str, Any]:
    client = None
    if not dry_run and model_settings.provider == "openai":
        client = openai_client_from_settings(model_settings)

    schema = {
        "type": "OBJECT",
        "properties": {
            "build_substitutes": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {"capability": {"type": "STRING"}},
                    "required": ["capability"],
                },
            },
            "license_sells": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "tech_id": {"type": "STRING"},
                        "min_price": {"type": "NUMBER"},
                    },
                    "required": ["tech_id", "min_price"],
                },
            },
            "license_buys": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "tech_id": {"type": "STRING"},
                        "max_price": {"type": "NUMBER"},
                    },
                    "required": ["tech_id", "max_price"],
                },
            },
            "fulfill_orders": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
            "commentary": {"type": "STRING"},
        },
    }

    modules_by_id = _module_index(modules)
    modules_by_capability = _capability_index(modules)
    orders_by_id = _order_index(orders)
    orders_by_firm = _orders_by_firm(orders)
    holdings_by_firm = _initial_holdings(modules)
    completed_substitutes_by_firm: Dict[str, Set[str]] = {firm.firm_id: set() for firm in firms}
    substitute_projects_by_firm: Dict[str, List[SubstituteProject]] = {
        firm.firm_id: [] for firm in firms
    }
    licensed_modules_by_firm: Dict[str, Set[str]] = {firm.firm_id: set() for firm in firms}
    reputation = {firm.firm_id: 0.0 for firm in firms}
    history_by_firm: Dict[str, List[Dict[str, Any]]] = {firm.firm_id: [] for firm in firms}
    last_round_deals: List[Dict[str, Any]] = []
    fulfilled_orders: Set[str] = set()
    started_orders: Set[str] = set()
    customer_value_captured = 0.0
    total_welfare = 0.0
    total_deal_volume = 0
    total_substitute_builds_started = 0
    total_substitute_builds_completed = 0
    trade_active_firm_rounds = 0
    round_logs: List[Dict[str, Any]] = []
    profits_prev: Dict[str, float] = {firm.firm_id: 0.0 for firm in firms}

    for round_index in range(1, rounds + 1):
        completed_substitutes = _complete_substitutes_for_round(
            current_round=round_index,
            substitute_projects_by_firm=substitute_projects_by_firm,
            completed_substitutes_by_firm=completed_substitutes_by_firm,
        )
        total_substitute_builds_completed += len(completed_substitutes)

        sorted_firms = sorted(
            firms,
            key=lambda firm: profits_prev.get(firm.firm_id, 0.0),
            reverse=True,
        )
        ranks = {firm.firm_id: index + 1 for index, firm in enumerate(sorted_firms)}

        plans: Dict[str, Dict[str, Any]] = {}
        usage_by_firm: Dict[str, Dict[str, int]] = {}
        round_profits: Dict[str, float] = {firm.firm_id: 0.0 for firm in firms}

        for firm in firms:
            metadata: Dict[str, Any] = {}
            user_payload = _firm_private_input(
                firm=firm,
                modules=modules,
                orders=orders_by_firm.get(firm.firm_id, []),
                holdings_by_firm=holdings_by_firm,
                completed_substitutes_by_firm=completed_substitutes_by_firm,
                substitute_projects_by_firm=substitute_projects_by_firm,
                reputation=reputation,
                last_round_deals=last_round_deals,
                history=history_by_firm[firm.firm_id],
                score=profits_prev.get(firm.firm_id, 0.0),
                rank=ranks.get(firm.firm_id, len(firms)),
                total_firms=len(firms),
                fulfilled_orders=fulfilled_orders,
            )
            data, raw_text = call_agent_json(
                client=client,
                model_settings=model_settings,
                system_prompt=firm.system_prompt,
                user_prompt=json.dumps(user_payload),
                response_schema=schema,
                dry_run=dry_run,
                metadata_sink=metadata,
            )
            build_substitutes, license_sells, license_buys, fulfill_orders, commentary = parse_firm_plan(
                data
            )
            plans[firm.firm_id] = {
                "build_substitutes": build_substitutes,
                "license_sells": license_sells,
                "license_buys": license_buys,
                "fulfill_orders": fulfill_orders,
                "commentary": commentary,
                "raw_text": raw_text,
            }
            if metadata.get("usage"):
                usage_by_firm[firm.firm_id] = metadata["usage"]

        trade_active_firm_rounds += sum(
            1
            for plan in plans.values()
            if plan["license_sells"] or plan["license_buys"]
        )

        substitute_builds_started: List[Dict[str, Any]] = []
        for firm in firms:
            firm_id = firm.firm_id
            available_capabilities = _available_capabilities_for_firm(
                firm_id=firm_id,
                holdings_by_firm=holdings_by_firm,
                completed_substitutes_by_firm=completed_substitutes_by_firm,
                modules_by_id=modules_by_id,
            )
            active_capabilities = {
                project.capability for project in substitute_projects_by_firm.get(firm_id, [])
            }
            for capability in plans[firm_id]["build_substitutes"]:
                module = modules_by_capability.get(capability)
                if module is None:
                    continue
                if module.owner == firm_id:
                    continue
                if capability in available_capabilities or capability in active_capabilities:
                    continue
                cost = module.substitute_costs[firm_id]
                project = SubstituteProject(
                    firm_id=firm_id,
                    capability=capability,
                    source_module_id=module.tech_id,
                    start_round=round_index,
                    ready_round=round_index + 1,
                    cost=cost,
                )
                substitute_projects_by_firm[firm_id].append(project)
                active_capabilities.add(capability)
                round_profits[firm_id] -= cost
                total_welfare -= cost
                total_substitute_builds_started += 1
                touched_orders = _orders_touched_by_capability(
                    firm_id=firm_id,
                    capability=capability,
                    orders_by_firm=orders_by_firm,
                    fulfilled_orders=fulfilled_orders,
                )
                started_orders.update(touched_orders)
                substitute_builds_started.append(
                    {
                        "firm_id": firm_id,
                        "capability": capability,
                        "source_module_id": module.tech_id,
                        "cost": round(cost, 4),
                        "ready_round": round_index + 1,
                        "touched_orders": touched_orders,
                    }
                )

        successful_deals: List[Dict[str, Any]] = []
        failed_deals: List[Dict[str, Any]] = []
        for module in modules:
            asks: List[Tuple[str, float]] = []
            bids: List[Tuple[str, float]] = []
            for firm_id, plan in plans.items():
                if firm_id == module.owner:
                    for ask in plan["license_sells"]:
                        if str(ask.get("tech_id") or "").strip() != module.tech_id:
                            continue
                        asks.append((firm_id, safe_float(ask.get("min_price"), 0.0) or 0.0))
                for bid in plan["license_buys"]:
                    if str(bid.get("tech_id") or "").strip() != module.tech_id:
                        continue
                    if firm_id == module.owner or module.tech_id in holdings_by_firm.get(firm_id, set()):
                        continue
                    bids.append((firm_id, safe_float(bid.get("max_price"), 0.0) or 0.0))
            if not asks or not bids:
                continue

            seller_id, ask_price = min(asks, key=lambda item: item[1])
            buyer_id, bid_price = max(bids, key=lambda item: item[1])
            if bid_price < ask_price:
                continue

            clearing_price = round((bid_price + ask_price) / 2.0, 4)
            if _deal_failure(
                seed=seed,
                round_index=round_index,
                tech_id=module.tech_id,
                seller_id=seller_id,
                buyer_id=buyer_id,
                quality=module.quality,
            ):
                reputation[seller_id] -= 2.0
                reputation[buyer_id] -= 1.0
                failed_deals.append(
                    {
                        "tech_id": module.tech_id,
                        "capability": module.capability,
                        "seller": seller_id,
                        "buyer": buyer_id,
                        "price": clearing_price,
                    }
                )
                continue

            holdings_by_firm.setdefault(buyer_id, set()).add(module.tech_id)
            licensed_modules_by_firm[buyer_id].add(module.tech_id)
            reputation[seller_id] += 1.0
            reputation[buyer_id] += 0.5

            buyer_profit = (
                -clearing_price
                - 0.25
                - module.integration_costs[buyer_id]
            )
            seller_profit = clearing_price - 0.25
            round_profits[buyer_id] += buyer_profit
            round_profits[seller_id] += seller_profit
            total_welfare -= module.integration_costs[buyer_id] + 0.5
            total_deal_volume += 1
            touched_orders = _orders_touched_by_capability(
                firm_id=buyer_id,
                capability=module.capability,
                orders_by_firm=orders_by_firm,
                fulfilled_orders=fulfilled_orders,
            )
            started_orders.update(touched_orders)
            successful_deals.append(
                {
                    "tech_id": module.tech_id,
                    "capability": module.capability,
                    "seller": seller_id,
                    "buyer": buyer_id,
                    "price": clearing_price,
                    "buyer_profit": round(buyer_profit, 4),
                    "seller_profit": round(seller_profit, 4),
                    "touched_orders": touched_orders,
                }
            )

        available_capabilities_after_trade = _available_capabilities_by_firm(
            firms=firms,
            holdings_by_firm=holdings_by_firm,
            completed_substitutes_by_firm=completed_substitutes_by_firm,
            modules_by_id=modules_by_id,
        )

        fulfillment_attempts_by_firm = {
            firm_id: list(plan["fulfill_orders"]) for firm_id, plan in plans.items() if plan["fulfill_orders"]
        }
        for order_ids in fulfillment_attempts_by_firm.values():
            started_orders.update(order_ids)

        successful_fulfillments, failed_fulfillments, round_customer_value = evaluate_fulfillment_attempts(
            orders_by_id=orders_by_id,
            available_capabilities_by_firm=available_capabilities_after_trade,
            fulfillment_attempts_by_firm=fulfillment_attempts_by_firm,
            fulfilled_orders=fulfilled_orders,
            current_round=round_index,
        )
        for fulfillment in successful_fulfillments:
            round_profits[fulfillment["firm_id"]] += fulfillment["delivery_value"]
        total_welfare += round_customer_value
        customer_value_captured += round_customer_value

        order_state_after = order_state_snapshot(
            orders=orders,
            available_capabilities_by_firm=available_capabilities_after_trade,
            started_orders=started_orders,
            fulfilled_orders=fulfilled_orders,
            substitute_projects_by_firm=substitute_projects_by_firm,
        )
        started_not_delivered = started_not_delivered_orders(order_state_after)

        round_usage = None
        if usage_by_firm:
            round_usage = {
                "calls": sum(item.get("calls", 0) for item in usage_by_firm.values()),
                "input_tokens": sum(item.get("input_tokens", 0) for item in usage_by_firm.values()),
                "output_tokens": sum(item.get("output_tokens", 0) for item in usage_by_firm.values()),
            }

        for firm in firms:
            history_by_firm[firm.firm_id].append(
                {
                    "round": round_index,
                    "profit": round(round_profits[firm.firm_id], 4),
                    "reputation": round(reputation[firm.firm_id], 4),
                    "held_modules": sorted(holdings_by_firm.get(firm.firm_id, set())),
                    "completed_substitutes": sorted(completed_substitutes_by_firm.get(firm.firm_id, set())),
                    "successful_deals": [
                        deal
                        for deal in successful_deals
                        if deal["buyer"] == firm.firm_id or deal["seller"] == firm.firm_id
                    ],
                    "successful_fulfillments": [
                        fulfillment
                        for fulfillment in successful_fulfillments
                        if fulfillment["firm_id"] == firm.firm_id
                    ],
                }
            )

        profits_prev = dict(round_profits)
        round_logs.append(
            {
                "round": round_index,
                "plans": {
                    firm_id: {
                        "build_substitutes": plan["build_substitutes"],
                        "license_sells": plan["license_sells"],
                        "license_buys": plan["license_buys"],
                        "fulfill_orders": plan["fulfill_orders"],
                        "commentary": plan["commentary"],
                    }
                    for firm_id, plan in plans.items()
                },
                "substitute_builds_started": substitute_builds_started,
                "substitute_builds_completed": completed_substitutes,
                "successful_deals": successful_deals,
                "failed_deals": failed_deals,
                "fulfillment_attempts": fulfillment_attempts_by_firm,
                "successful_fulfillments": successful_fulfillments,
                "failed_fulfillments": failed_fulfillments,
                "order_state": order_state_after,
                "started_not_delivered": started_not_delivered,
                "holdings_by_firm": {
                    firm_id: sorted(holdings) for firm_id, holdings in holdings_by_firm.items()
                },
                "completed_substitutes_by_firm": {
                    firm_id: sorted(items)
                    for firm_id, items in completed_substitutes_by_firm.items()
                },
                "active_substitute_projects": {
                    firm_id: [
                        {
                            "capability": project.capability,
                            "ready_round": project.ready_round,
                            "cost": round(project.cost, 4),
                        }
                        for project in projects
                    ]
                    for firm_id, projects in substitute_projects_by_firm.items()
                },
                "self_initiated_trade_firms": sorted(
                    firm_id
                    for firm_id, plan in plans.items()
                    if plan["license_sells"] or plan["license_buys"]
                ),
                "reputation": {firm_id: round(value, 4) for firm_id, value in reputation.items()},
                "profits": {firm_id: round(value, 4) for firm_id, value in round_profits.items()},
                "welfare": round(total_welfare, 4),
                "customer_value_captured": round(customer_value_captured, 4),
                "codex_usage": round_usage,
            }
        )
        last_round_deals = successful_deals

    final_available_capabilities = _available_capabilities_by_firm(
        firms=firms,
        holdings_by_firm=holdings_by_firm,
        completed_substitutes_by_firm=completed_substitutes_by_firm,
        modules_by_id=modules_by_id,
    )
    final_order_state = order_state_snapshot(
        orders=orders,
        available_capabilities_by_firm=final_available_capabilities,
        started_orders=started_orders,
        fulfilled_orders=fulfilled_orders,
        substitute_projects_by_firm=substitute_projects_by_firm,
    )
    summary = build_summary(
        orders=orders,
        order_state=final_order_state,
        total_deal_volume=total_deal_volume,
        total_substitute_builds_started=total_substitute_builds_started,
        total_substitute_builds_completed=total_substitute_builds_completed,
        total_welfare=total_welfare,
        customer_value_captured=customer_value_captured,
        trade_active_firm_rounds=trade_active_firm_rounds,
        total_firm_rounds=len(firms) * rounds,
    )

    return {
        "summary": summary,
        "round_logs": round_logs,
        "orders": [
            {
                "order_id": order.order_id,
                "target_firm": order.target_firm,
                "customer_brief": order.customer_brief,
                "delivery_value": order.delivery_value,
                "deadline_round": order.deadline_round,
                "required_capabilities": list(order.required_capabilities),
            }
            for order in orders
        ],
        "modules": [
            {
                "tech_id": module.tech_id,
                "owner": module.owner,
                "capability": module.capability,
                "module_name": module.module_name,
                "quality": module.quality,
            }
            for module in modules
        ],
        "licensed_modules_by_firm": {
            firm_id: sorted(items) for firm_id, items in licensed_modules_by_firm.items()
        },
        "completed_substitutes_by_firm": {
            firm_id: sorted(items) for firm_id, items in completed_substitutes_by_firm.items()
        },
    }
