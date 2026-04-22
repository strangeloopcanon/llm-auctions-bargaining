from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Sequence, Set, Tuple

from ..agents import call_agent_json, openai_client_from_settings
from ..protocols import safe_float
from ..settings import ModelSettings
from .models import CustomerOrder, Firm, Module, SubstituteProject


SCENARIO_CONFIGS: Dict[str, Dict[str, Any]] = {
    "partner_directory": {
        "directory_style": "exact",
        "request_match_threshold": 2,
        "coordination_cost": 0.5,
    },
    "opaque_directory": {
        "directory_style": "opaque",
        "request_match_threshold": 3,
        "coordination_cost": 0.75,
    },
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "your",
    "their",
    "this",
    "that",
    "our",
    "will",
    "would",
}


def parse_firm_plan(
    raw: Dict[str, Any],
) -> Tuple[List[str], List[Dict[str, Any]], List[str], str]:
    internal_projects_raw = raw.get("internal_projects") if isinstance(raw, dict) else []
    partner_requests_raw = raw.get("partner_requests") if isinstance(raw, dict) else []
    fulfill_orders_raw = raw.get("fulfill_orders") if isinstance(raw, dict) else []
    commentary = raw.get("commentary", "") if isinstance(raw, dict) else ""

    internal_projects: List[str] = []
    for item in list(internal_projects_raw or []):
        if isinstance(item, dict):
            objective = str(item.get("objective") or "").strip()
        else:
            objective = str(item).strip()
        if objective:
            internal_projects.append(objective)

    partner_requests: List[Dict[str, Any]] = []
    for item in list(partner_requests_raw or []):
        if not isinstance(item, dict):
            continue
        to_firm = str(item.get("to_firm") or "").strip()
        request = str(item.get("request") or "").strip()
        cash_offer = safe_float(item.get("cash_offer"), None)
        note = str(item.get("note") or "").strip()
        if not to_firm or not request or cash_offer is None:
            continue
        partner_requests.append(
            {
                "to_firm": to_firm,
                "request": request,
                "cash_offer": float(cash_offer),
                "note": note,
            }
        )

    fulfill_orders = [str(item).strip() for item in list(fulfill_orders_raw or []) if str(item).strip()]
    return internal_projects, partner_requests, fulfill_orders, commentary


def _tokenize(text: str) -> Set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(token) >= 3 and token not in STOPWORDS
    }


def _module_lookup_text(module: Module) -> str:
    return " ".join(
        [
            module.capability,
            module.module_name,
            module.description,
            module.order_phrase,
        ]
    )


def _module_match_score(text: str, module: Module) -> int:
    normalized_text = str(text or "").strip().lower()
    if not normalized_text:
        return 0
    overlap = len(_tokenize(normalized_text) & _tokenize(_module_lookup_text(module)))
    if module.capability.lower() in normalized_text:
        overlap += 3
    if module.module_name.lower() in normalized_text:
        overlap += 2
    if module.order_phrase.lower() in normalized_text:
        overlap += 2
    return overlap


def _resolve_module_from_text(
    *,
    text: str,
    candidate_modules: Sequence[Module],
    min_score: int,
) -> Module | None:
    best_module: Module | None = None
    best_score = 0
    second_best = 0
    for module in candidate_modules:
        score = _module_match_score(text, module)
        if score > best_score:
            second_best = best_score
            best_score = score
            best_module = module
            continue
        if score > second_best:
            second_best = score
    if best_module is None or best_score < min_score:
        return None
    if best_score == second_best:
        return None
    return best_module


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
    total_partner_requests: int,
    total_substitute_builds_started: int,
    total_substitute_builds_completed: int,
    total_welfare: float,
    customer_value_captured: float,
    partner_request_firm_rounds: int,
    total_firm_rounds: int,
) -> Dict[str, Any]:
    total_orders = len(orders)
    fulfilled = sum(1 for state in order_state.values() if state.get("fulfilled"))
    started_not_delivered = started_not_delivered_orders(order_state)
    partner_request_rate = (
        round(partner_request_firm_rounds / total_firm_rounds, 4) if total_firm_rounds else 0.0
    )
    return {
        "total_orders": total_orders,
        "orders_fulfilled": fulfilled,
        "fulfillment_rate": round(fulfilled / total_orders, 4) if total_orders else 0.0,
        "started_not_delivered_count": len(started_not_delivered),
        "started_not_delivered_orders": started_not_delivered,
        "partner_request_rate": partner_request_rate,
        "self_initiated_trade_rate": partner_request_rate,
        "partner_requests_submitted": total_partner_requests,
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


def _modules_by_owner(modules: Sequence[Module]) -> Dict[str, List[Module]]:
    grouped: Dict[str, List[Module]] = {}
    for module in modules:
        grouped.setdefault(module.owner, []).append(module)
    return grouped


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


def _strength_briefs(
    *,
    firm_id: str,
    holdings_by_firm: Dict[str, Set[str]],
    completed_substitutes_by_firm: Dict[str, Set[str]],
    modules_by_id: Dict[str, Module],
    modules_by_capability: Dict[str, Module],
) -> List[Dict[str, str]]:
    strengths: List[Dict[str, str]] = []
    for tech_id in sorted(holdings_by_firm.get(firm_id, set())):
        module = modules_by_id.get(tech_id)
        if module is None:
            continue
        strengths.append({"brief": module.order_phrase, "details": module.description})
    for capability in sorted(completed_substitutes_by_firm.get(firm_id, set())):
        module = modules_by_capability.get(capability)
        if module is None:
            continue
        strengths.append(
            {
                "brief": module.order_phrase,
                "details": f"In-house substitute for {module.description.lower()}",
            }
        )
    return strengths


def _counterparty_directory(
    *,
    firm_id: str,
    firms: Sequence[Firm],
    modules_by_owner: Dict[str, List[Module]],
    reputation: Dict[str, float],
    scenario_name: str,
) -> List[Dict[str, Any]]:
    style = SCENARIO_CONFIGS[scenario_name]["directory_style"]
    entries: List[Dict[str, Any]] = []
    for other in firms:
        if other.firm_id == firm_id:
            continue
        modules = modules_by_owner.get(other.firm_id, [])
        if style == "exact":
            entries.append(
                {
                    "firm_id": other.firm_id,
                    "public_summary": (
                        f"{other.firm_id} is known for "
                        + ", ".join(module.order_phrase for module in modules[:3])
                        + "."
                    ),
                    "known_strengths": [module.order_phrase for module in modules],
                    "reputation_score": round(reputation.get(other.firm_id, 0.0), 4),
                }
            )
            continue
        entries.append(
            {
                "firm_id": other.firm_id,
                "public_summary": " ".join(module.description for module in modules[:2]),
                "recent_delivery_focus": [module.order_phrase for module in modules[:2]],
                "reputation_score": round(reputation.get(other.firm_id, 0.0), 4),
            }
        )
    return entries


def _internal_project_options(
    *,
    firm_id: str,
    modules: Sequence[Module],
    holdings_by_firm: Dict[str, Set[str]],
    completed_substitutes_by_firm: Dict[str, Set[str]],
    modules_by_id: Dict[str, Module],
) -> List[Dict[str, Any]]:
    available_capabilities = _available_capabilities_for_firm(
        firm_id=firm_id,
        holdings_by_firm=holdings_by_firm,
        completed_substitutes_by_firm=completed_substitutes_by_firm,
        modules_by_id=modules_by_id,
    )
    options: List[Dict[str, Any]] = []
    for module in modules:
        if module.capability in available_capabilities:
            continue
        options.append(
            {
                "project_brief": module.order_phrase,
                "details": module.description,
                "estimated_cost": module.substitute_costs.get(firm_id, 0.0),
                "ready_next_round": True,
            }
        )
    return options


def _public_last_round_deals(last_round_deals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "seller": deal["seller"],
            "buyer": deal["buyer"],
            "public_need": deal["public_need"],
            "price": deal["price"],
        }
        for deal in last_round_deals
    ]


def _firm_private_input(
    *,
    firm: Firm,
    firms: Sequence[Firm],
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
    scenario_name: str,
) -> Dict[str, Any]:
    modules_by_id = _module_index(modules)
    modules_by_capability = _capability_index(modules)
    modules_by_owner = _modules_by_owner(modules)

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

    active_projects = []
    for project in substitute_projects_by_firm.get(firm.firm_id, []):
        module = modules_by_capability.get(project.capability)
        if module is None:
            continue
        active_projects.append(
            {
                "project_brief": module.order_phrase,
                "details": module.description,
                "ready_round": project.ready_round,
                "cost": project.cost,
            }
        )

    return {
        "firm_id": firm.firm_id,
        "controlled_strengths": _strength_briefs(
            firm_id=firm.firm_id,
            holdings_by_firm=holdings_by_firm,
            completed_substitutes_by_firm=completed_substitutes_by_firm,
            modules_by_id=modules_by_id,
            modules_by_capability=modules_by_capability,
        ),
        "active_internal_projects": active_projects,
        "internal_project_options": _internal_project_options(
            firm_id=firm.firm_id,
            modules=modules,
            holdings_by_firm=holdings_by_firm,
            completed_substitutes_by_firm=completed_substitutes_by_firm,
            modules_by_id=modules_by_id,
        ),
        "customer_orders": customer_orders,
        "partner_directory": _counterparty_directory(
            firm_id=firm.firm_id,
            firms=firms,
            modules_by_owner=modules_by_owner,
            reputation=reputation,
            scenario_name=scenario_name,
        ),
        "coordination_cost_per_request": SCENARIO_CONFIGS[scenario_name]["coordination_cost"],
        "reputation_score": round(reputation.get(firm.firm_id, 0.0), 4),
        "last_round_outside_arrangements": _public_last_round_deals(last_round_deals),
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
    scenario: str = "partner_directory",
) -> Dict[str, Any]:
    if scenario not in SCENARIO_CONFIGS:
        raise ValueError(f"Unknown completion-market scenario: {scenario}")

    client = None
    if not dry_run and model_settings.provider == "openai":
        client = openai_client_from_settings(model_settings)

    schema = {
        "type": "OBJECT",
        "properties": {
            "internal_projects": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {"objective": {"type": "STRING"}},
                    "required": ["objective"],
                },
            },
            "partner_requests": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "to_firm": {"type": "STRING"},
                        "request": {"type": "STRING"},
                        "cash_offer": {"type": "NUMBER"},
                        "note": {"type": "STRING"},
                    },
                    "required": ["to_firm", "request", "cash_offer", "note"],
                },
            },
            "fulfill_orders": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
            "commentary": {"type": "STRING"},
        },
    }

    scenario_config = SCENARIO_CONFIGS[scenario]
    modules_by_id = _module_index(modules)
    modules_by_owner = _modules_by_owner(modules)
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
    total_partner_requests = 0
    total_substitute_builds_started = 0
    total_substitute_builds_completed = 0
    partner_request_firm_rounds = 0
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
                firms=firms,
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
                scenario_name=scenario,
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
            internal_projects, partner_requests, fulfill_orders, commentary = parse_firm_plan(data)
            plans[firm.firm_id] = {
                "internal_projects": internal_projects,
                "partner_requests": partner_requests,
                "fulfill_orders": fulfill_orders,
                "commentary": commentary,
                "raw_text": raw_text,
            }
            if metadata.get("usage"):
                usage_by_firm[firm.firm_id] = metadata["usage"]

        partner_request_firm_rounds += sum(
            1 for plan in plans.values() if plan["partner_requests"]
        )
        total_partner_requests += sum(len(plan["partner_requests"]) for plan in plans.values())

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
            for objective in plans[firm_id]["internal_projects"]:
                module = _resolve_module_from_text(
                    text=objective,
                    candidate_modules=modules,
                    min_score=2,
                )
                if module is None:
                    continue
                if module.owner == firm_id:
                    continue
                if module.capability in available_capabilities or module.capability in active_capabilities:
                    continue
                cost = module.substitute_costs[firm_id]
                project = SubstituteProject(
                    firm_id=firm_id,
                    capability=module.capability,
                    source_module_id=module.tech_id,
                    start_round=round_index,
                    ready_round=round_index + 1,
                    cost=cost,
                )
                substitute_projects_by_firm[firm_id].append(project)
                active_capabilities.add(module.capability)
                round_profits[firm_id] -= cost
                total_welfare -= cost
                total_substitute_builds_started += 1
                touched_orders = _orders_touched_by_capability(
                    firm_id=firm_id,
                    capability=module.capability,
                    orders_by_firm=orders_by_firm,
                    fulfilled_orders=fulfilled_orders,
                )
                started_orders.update(touched_orders)
                substitute_builds_started.append(
                    {
                        "firm_id": firm_id,
                        "objective": objective,
                        "capability": module.capability,
                        "public_need": module.order_phrase,
                        "source_module_id": module.tech_id,
                        "cost": round(cost, 4),
                        "ready_round": round_index + 1,
                        "touched_orders": touched_orders,
                    }
                )

        successful_deals: List[Dict[str, Any]] = []
        failed_deals: List[Dict[str, Any]] = []
        unresolved_partner_requests: List[Dict[str, Any]] = []
        seen_requests: Set[Tuple[str, str, str]] = set()
        for buyer_id, plan in plans.items():
            for request in plan["partner_requests"]:
                request_key = (buyer_id, request["to_firm"], request["request"])
                if request_key in seen_requests:
                    continue
                seen_requests.add(request_key)

                target_firm = request["to_firm"]
                if target_firm == buyer_id:
                    unresolved_partner_requests.append(
                        {
                            "buyer": buyer_id,
                            "to_firm": target_firm,
                            "request": request["request"],
                            "reason": "self_request",
                        }
                    )
                    continue
                candidate_modules = modules_by_owner.get(target_firm, [])
                if not candidate_modules:
                    unresolved_partner_requests.append(
                        {
                            "buyer": buyer_id,
                            "to_firm": target_firm,
                            "request": request["request"],
                            "reason": "unknown_target_firm",
                        }
                    )
                    continue
                module = _resolve_module_from_text(
                    text=request["request"],
                    candidate_modules=candidate_modules,
                    min_score=scenario_config["request_match_threshold"],
                )
                if module is None:
                    unresolved_partner_requests.append(
                        {
                            "buyer": buyer_id,
                            "to_firm": target_firm,
                            "request": request["request"],
                            "reason": "unresolved_request",
                        }
                    )
                    continue
                if module.tech_id in holdings_by_firm.get(buyer_id, set()):
                    unresolved_partner_requests.append(
                        {
                            "buyer": buyer_id,
                            "to_firm": target_firm,
                            "request": request["request"],
                            "matched_capability": module.capability,
                            "reason": "already_controlled",
                        }
                    )
                    continue

                reservation_price = round(module.reference_license_price + 0.25, 4)
                if request["cash_offer"] < reservation_price:
                    unresolved_partner_requests.append(
                        {
                            "buyer": buyer_id,
                            "to_firm": target_firm,
                            "request": request["request"],
                            "matched_capability": module.capability,
                            "cash_offer": round(request["cash_offer"], 4),
                            "reservation_price": reservation_price,
                            "reason": "offer_too_low",
                        }
                    )
                    continue

                clearing_price = round((request["cash_offer"] + reservation_price) / 2.0, 4)
                if _deal_failure(
                    seed=seed,
                    round_index=round_index,
                    tech_id=module.tech_id,
                    seller_id=target_firm,
                    buyer_id=buyer_id,
                    quality=module.quality,
                ):
                    reputation[target_firm] -= 2.0
                    reputation[buyer_id] -= 1.0
                    failed_deals.append(
                        {
                            "tech_id": module.tech_id,
                            "capability": module.capability,
                            "public_need": module.order_phrase,
                            "seller": target_firm,
                            "buyer": buyer_id,
                            "price": clearing_price,
                        }
                    )
                    continue

                holdings_by_firm.setdefault(buyer_id, set()).add(module.tech_id)
                licensed_modules_by_firm[buyer_id].add(module.tech_id)
                reputation[target_firm] += 1.0
                reputation[buyer_id] += 0.5

                coordination_cost = float(scenario_config["coordination_cost"])
                buyer_profit = -clearing_price - coordination_cost - module.integration_costs[buyer_id]
                seller_profit = clearing_price - coordination_cost
                round_profits[buyer_id] += buyer_profit
                round_profits[target_firm] += seller_profit
                total_welfare -= module.integration_costs[buyer_id] + (2.0 * coordination_cost)
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
                        "public_need": module.order_phrase,
                        "seller": target_firm,
                        "buyer": buyer_id,
                        "price": clearing_price,
                        "buyer_profit": round(buyer_profit, 4),
                        "seller_profit": round(seller_profit, 4),
                        "request": request["request"],
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
                    "outside_arrangements": [
                        {
                            "counterparty": (
                                deal["seller"] if deal["buyer"] == firm.firm_id else deal["buyer"]
                            ),
                            "public_need": deal["public_need"],
                            "price": deal["price"],
                        }
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
                        "internal_projects": plan["internal_projects"],
                        "partner_requests": plan["partner_requests"],
                        "fulfill_orders": plan["fulfill_orders"],
                        "commentary": plan["commentary"],
                    }
                    for firm_id, plan in plans.items()
                },
                "substitute_builds_started": substitute_builds_started,
                "substitute_builds_completed": completed_substitutes,
                "successful_deals": successful_deals,
                "failed_deals": failed_deals,
                "unresolved_partner_requests": unresolved_partner_requests,
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
                "partner_request_firms": sorted(
                    firm_id
                    for firm_id, plan in plans.items()
                    if plan["partner_requests"]
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
        total_partner_requests=total_partner_requests,
        total_substitute_builds_started=total_substitute_builds_started,
        total_substitute_builds_completed=total_substitute_builds_completed,
        total_welfare=total_welfare,
        customer_value_captured=customer_value_captured,
        partner_request_firm_rounds=partner_request_firm_rounds,
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
