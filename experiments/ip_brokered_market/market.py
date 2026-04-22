from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Sequence, Set, Tuple

from ..agents import call_agent_json, openai_client_from_settings
from ..protocols import safe_float
from ..settings import ModelSettings
from .models import BoardNotice, BoardReply, CustomerOrder, Firm, InternalProject, Module


STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "this",
    "to",
    "we",
    "with",
    "your",
}

OFFER_KEYWORDS = (
    "available",
    "can provide",
    "can supply",
    "can support",
    "can deliver",
    "offering",
    "offer",
    "provide",
    "supply",
)


def parse_firm_plan(
    raw: Dict[str, Any],
) -> Tuple[List[str], List[Dict[str, str]], List[Dict[str, Any]], List[str], str]:
    if not isinstance(raw, dict):
        return [], [], [], [], ""

    internal_projects: List[str] = []
    for item in list(raw.get("internal_projects") or []):
        if isinstance(item, dict):
            objective = str(item.get("objective") or "").strip()
        else:
            objective = str(item).strip()
        if objective:
            internal_projects.append(objective)

    public_notices: List[Dict[str, str]] = []
    for item in list(raw.get("public_notices") or []):
        if not isinstance(item, dict):
            continue
        headline = str(item.get("headline") or "").strip()
        body = str(item.get("body") or "").strip()
        cash_terms = str(item.get("cash_terms") or "").strip()
        if not headline and not body:
            continue
        public_notices.append(
            {
                "headline": headline,
                "body": body,
                "cash_terms": cash_terms,
            }
        )

    reply_notices: List[Dict[str, Any]] = []
    for item in list(raw.get("reply_notices") or []):
        if not isinstance(item, dict):
            continue
        notice_id = str(item.get("notice_id") or "").strip()
        body = str(item.get("body") or "").strip()
        cash_offer = safe_float(item.get("cash_offer"), None)
        if not notice_id or not body or cash_offer is None:
            continue
        reply_notices.append(
            {
                "notice_id": notice_id,
                "body": body,
                "cash_offer": float(cash_offer),
            }
        )

    fulfill_orders = [
        str(item).strip()
        for item in list(raw.get("fulfill_orders") or [])
        if str(item).strip()
    ]
    commentary = str(raw.get("commentary") or "").strip()
    return internal_projects, public_notices, reply_notices, fulfill_orders, commentary


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
    normalized = str(text or "").strip().lower()
    if not normalized:
        return 0
    overlap = len(_tokenize(normalized) & _tokenize(_module_lookup_text(module)))
    if module.capability.lower() in normalized:
        overlap += 3
    if module.module_name.lower() in normalized:
        overlap += 2
    if module.order_phrase.lower() in normalized:
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
    completed_internal_capabilities_by_firm: Dict[str, Set[str]],
    modules_by_id: Dict[str, Module],
) -> Set[str]:
    capabilities = {
        modules_by_id[tech_id].capability
        for tech_id in holdings_by_firm.get(firm_id, set())
        if tech_id in modules_by_id
    }
    capabilities.update(completed_internal_capabilities_by_firm.get(firm_id, set()))
    return capabilities


def _available_capabilities_by_firm(
    *,
    firms: Sequence[Firm],
    holdings_by_firm: Dict[str, Set[str]],
    completed_internal_capabilities_by_firm: Dict[str, Set[str]],
    modules_by_id: Dict[str, Module],
) -> Dict[str, Set[str]]:
    return {
        firm.firm_id: _available_capabilities_for_firm(
            firm_id=firm.firm_id,
            holdings_by_firm=holdings_by_firm,
            completed_internal_capabilities_by_firm=completed_internal_capabilities_by_firm,
            modules_by_id=modules_by_id,
        )
        for firm in firms
    }


def _strength_briefs(
    *,
    firm_id: str,
    holdings_by_firm: Dict[str, Set[str]],
    completed_internal_capabilities_by_firm: Dict[str, Set[str]],
    modules_by_id: Dict[str, Module],
    modules_by_capability: Dict[str, Module],
) -> List[Dict[str, str]]:
    strengths: List[Dict[str, str]] = []
    for tech_id in sorted(holdings_by_firm.get(firm_id, set())):
        module = modules_by_id.get(tech_id)
        if module is None:
            continue
        strengths.append({"brief": module.order_phrase, "details": module.description})
    for capability in sorted(completed_internal_capabilities_by_firm.get(firm_id, set())):
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


def _internal_project_options(
    *,
    firm_id: str,
    modules: Sequence[Module],
    holdings_by_firm: Dict[str, Set[str]],
    completed_internal_capabilities_by_firm: Dict[str, Set[str]],
    internal_projects_by_firm: Dict[str, List[InternalProject]],
    modules_by_id: Dict[str, Module],
    current_round: int,
) -> List[Dict[str, Any]]:
    available_capabilities = _available_capabilities_for_firm(
        firm_id=firm_id,
        holdings_by_firm=holdings_by_firm,
        completed_internal_capabilities_by_firm=completed_internal_capabilities_by_firm,
        modules_by_id=modules_by_id,
    )
    active_capabilities = {
        project.capability for project in internal_projects_by_firm.get(firm_id, [])
    }
    options: List[Dict[str, Any]] = []
    for module in modules:
        if module.capability in available_capabilities:
            continue
        if module.capability in active_capabilities:
            continue
        options.append(
            {
                "project_brief": module.order_phrase,
                "details": module.description,
                "estimated_cost": module.substitute_costs.get(firm_id, 0.0),
                "ready_round": current_round + 2,
            }
        )
    return options


def _board_notice_to_dict(notice: BoardNotice) -> Dict[str, Any]:
    return {
        "notice_id": notice.notice_id,
        "author": notice.author,
        "headline": notice.headline,
        "body": notice.body,
        "cash_terms": notice.cash_terms,
        "round_posted": notice.round_posted,
    }


def _board_reply_to_dict(reply: BoardReply) -> Dict[str, Any]:
    return {
        "reply_id": reply.reply_id,
        "notice_id": reply.notice_id,
        "author": reply.author,
        "body": reply.body,
        "cash_offer": round(reply.cash_offer, 4),
        "round_posted": reply.round_posted,
    }


def _public_reputation(
    *,
    firms: Sequence[Firm],
    reputation: Dict[str, float],
) -> List[Dict[str, Any]]:
    return [
        {
            "firm_id": firm.firm_id,
            "reputation_score": round(reputation.get(firm.firm_id, 0.0), 4),
        }
        for firm in firms
    ]


def _firm_private_input(
    *,
    firm: Firm,
    firms: Sequence[Firm],
    modules: Sequence[Module],
    orders: Sequence[CustomerOrder],
    holdings_by_firm: Dict[str, Set[str]],
    completed_internal_capabilities_by_firm: Dict[str, Set[str]],
    internal_projects_by_firm: Dict[str, List[InternalProject]],
    visible_notices: Sequence[BoardNotice],
    visible_replies: Sequence[BoardReply],
    reputation: Dict[str, float],
    last_round_public_outcomes: Sequence[Dict[str, Any]],
    history: List[Dict[str, Any]],
    score: float,
    rank: int,
    total_firms: int,
    fulfilled_orders: Set[str],
    current_round: int,
    total_rounds: int,
) -> Dict[str, Any]:
    modules_by_id = _module_index(modules)
    modules_by_capability = _capability_index(modules)

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
    for project in internal_projects_by_firm.get(firm.firm_id, []):
        module = modules_by_capability.get(project.capability)
        if module is None:
            continue
        active_projects.append(
            {
                "project_brief": module.order_phrase,
                "details": module.description,
                "ready_round": project.ready_round,
                "cost": round(project.cost, 4),
            }
        )

    return {
        "firm_id": firm.firm_id,
        "current_round": current_round,
        "total_rounds": total_rounds,
        "controlled_strengths": _strength_briefs(
            firm_id=firm.firm_id,
            holdings_by_firm=holdings_by_firm,
            completed_internal_capabilities_by_firm=completed_internal_capabilities_by_firm,
            modules_by_id=modules_by_id,
            modules_by_capability=modules_by_capability,
        ),
        "active_internal_projects": active_projects,
        "internal_project_options": _internal_project_options(
            firm_id=firm.firm_id,
            modules=modules,
            holdings_by_firm=holdings_by_firm,
            completed_internal_capabilities_by_firm=completed_internal_capabilities_by_firm,
            internal_projects_by_firm=internal_projects_by_firm,
            modules_by_id=modules_by_id,
            current_round=current_round,
        ),
        "customer_orders": customer_orders,
        "public_board": {
            "visible_notices": [_board_notice_to_dict(notice) for notice in visible_notices],
            "visible_replies": [_board_reply_to_dict(reply) for reply in visible_replies],
        },
        "public_reputation": _public_reputation(firms=firms, reputation=reputation),
        "last_round_public_outcomes": list(last_round_public_outcomes),
        "recent_history": history[-3:],
        "your_score_last_round": round(score, 4),
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


def order_state_snapshot(
    *,
    orders: Sequence[CustomerOrder],
    available_capabilities_by_firm: Dict[str, Set[str]],
    started_orders: Set[str],
    fulfilled_orders: Set[str],
    internal_projects_by_firm: Dict[str, List[InternalProject]],
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
        active_projects = sorted(
            project.capability
            for project in internal_projects_by_firm.get(order.target_firm, [])
            if project.capability in order.required_capabilities
        )
        snapshot[order.order_id] = {
            "target_firm": order.target_firm,
            "required_capabilities": list(order.required_capabilities),
            "covered_capabilities": covered_capabilities,
            "missing_capabilities": missing_capabilities,
            "active_internal_projects": active_projects,
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


def _complete_internal_projects_for_round(
    *,
    current_round: int,
    internal_projects_by_firm: Dict[str, List[InternalProject]],
    completed_internal_capabilities_by_firm: Dict[str, Set[str]],
) -> List[Dict[str, Any]]:
    completed: List[Dict[str, Any]] = []
    for firm_id, projects in internal_projects_by_firm.items():
        ready_projects = [project for project in projects if project.ready_round <= current_round]
        if not ready_projects:
            continue
        internal_projects_by_firm[firm_id] = [
            project for project in projects if project.ready_round > current_round
        ]
        for project in ready_projects:
            completed_internal_capabilities_by_firm.setdefault(firm_id, set()).add(project.capability)
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


def _notice_text(notice: BoardNotice) -> str:
    return " ".join([notice.headline, notice.body, notice.cash_terms]).strip()


def _notice_is_offer_like(notice: BoardNotice) -> bool:
    text = _notice_text(notice).lower()
    if not text:
        return False
    return any(keyword in text for keyword in OFFER_KEYWORDS)


def _public_outcomes(
    *,
    successful_deals: Sequence[Dict[str, Any]],
    failed_deals: Sequence[Dict[str, Any]],
    successful_fulfillments: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    outcomes: List[Dict[str, Any]] = []
    for deal in successful_deals:
        outcomes.append(
            {
                "kind": "deal",
                "seller": deal["seller"],
                "buyer": deal["buyer"],
                "headline": deal["headline"],
                "capability": deal["capability"],
                "price": deal["price"],
            }
        )
    for deal in failed_deals:
        outcomes.append(
            {
                "kind": "failed_arrangement",
                "seller": deal["seller"],
                "buyer": deal["buyer"],
                "headline": deal["headline"],
                "reason": deal["reason"],
            }
        )
    for fulfillment in successful_fulfillments:
        outcomes.append(
            {
                "kind": "delivery",
                "firm_id": fulfillment["firm_id"],
                "order_id": fulfillment["order_id"],
                "delivery_value": fulfillment["delivery_value"],
            }
        )
    return outcomes


def _round_usage(usage_by_firm: Dict[str, Dict[str, int]]) -> Dict[str, int] | None:
    if not usage_by_firm:
        return None
    return {
        "calls": sum(item.get("calls", 0) for item in usage_by_firm.values()),
        "input_tokens": sum(item.get("input_tokens", 0) for item in usage_by_firm.values()),
        "output_tokens": sum(item.get("output_tokens", 0) for item in usage_by_firm.values()),
    }


def build_summary(
    *,
    orders: Sequence[CustomerOrder],
    order_state: Dict[str, Dict[str, Any]],
    brokered_deal_volume: int,
    public_notices_posted: int,
    replies_posted: int,
    board_use_firm_rounds: int,
    notice_post_firm_rounds: int,
    reply_firm_rounds: int,
    total_firm_rounds: int,
    internal_projects_started: int,
    internal_projects_completed: int,
    total_welfare: float,
    customer_value_captured: float,
) -> Dict[str, Any]:
    total_orders = len(orders)
    fulfilled = sum(1 for state in order_state.values() if state.get("fulfilled"))
    started_not_delivered = started_not_delivered_orders(order_state)
    fulfillment_rate = round(fulfilled / total_orders, 4) if total_orders else 0.0
    notice_post_rate = (
        round(notice_post_firm_rounds / total_firm_rounds, 4) if total_firm_rounds else 0.0
    )
    reply_rate = round(reply_firm_rounds / total_firm_rounds, 4) if total_firm_rounds else 0.0
    board_use_rate = (
        round(board_use_firm_rounds / total_firm_rounds, 4) if total_firm_rounds else 0.0
    )
    return {
        "total_orders": total_orders,
        "orders_fulfilled": fulfilled,
        "fulfillment_rate": fulfillment_rate,
        "started_not_delivered_count": len(started_not_delivered),
        "started_not_delivered_orders": started_not_delivered,
        "notice_post_rate": notice_post_rate,
        "reply_rate": reply_rate,
        "board_use_rate": board_use_rate,
        "public_notices_posted": public_notices_posted,
        "reply_notices_posted": replies_posted,
        "brokered_deal_volume": brokered_deal_volume,
        "deal_volume": brokered_deal_volume,
        "internal_projects_started": internal_projects_started,
        "internal_projects_completed": internal_projects_completed,
        "welfare": round(total_welfare, 4),
        "customer_value_captured": round(customer_value_captured, 4),
    }


def run_brokered_market(
    model_settings: ModelSettings,
    firms: Sequence[Firm],
    modules: Sequence[Module],
    orders: Sequence[CustomerOrder],
    *,
    dry_run: bool = False,
    rounds: int = 3,
    seed: int = 0,
) -> Dict[str, Any]:
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
            "public_notices": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "headline": {"type": "STRING"},
                        "body": {"type": "STRING"},
                        "cash_terms": {"type": "STRING"},
                    },
                    "required": ["headline", "body", "cash_terms"],
                },
            },
            "reply_notices": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "notice_id": {"type": "STRING"},
                        "body": {"type": "STRING"},
                        "cash_offer": {"type": "NUMBER"},
                    },
                    "required": ["notice_id", "body", "cash_offer"],
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
    modules_by_owner = _modules_by_owner(modules)
    orders_by_id = _order_index(orders)
    orders_by_firm = _orders_by_firm(orders)
    holdings_by_firm = _initial_holdings(modules)
    completed_internal_capabilities_by_firm: Dict[str, Set[str]] = {
        firm.firm_id: set() for firm in firms
    }
    internal_projects_by_firm: Dict[str, List[InternalProject]] = {
        firm.firm_id: [] for firm in firms
    }
    brokered_modules_by_firm: Dict[str, Set[str]] = {firm.firm_id: set() for firm in firms}
    reputation = {firm.firm_id: 0.0 for firm in firms}
    history_by_firm: Dict[str, List[Dict[str, Any]]] = {firm.firm_id: [] for firm in firms}
    visible_notices: List[BoardNotice] = []
    visible_replies: List[BoardReply] = []
    pending_notices: List[BoardNotice] = []
    pending_replies: List[BoardReply] = []
    last_round_public_outcomes: List[Dict[str, Any]] = []
    fulfilled_orders: Set[str] = set()
    started_orders: Set[str] = set()
    notice_counter = 1
    reply_counter = 1
    customer_value_captured = 0.0
    total_welfare = 0.0
    total_deal_volume = 0
    total_notices_posted = 0
    total_replies_posted = 0
    total_internal_projects_started = 0
    total_internal_projects_completed = 0
    board_use_firm_rounds = 0
    notice_post_firm_rounds = 0
    reply_firm_rounds = 0
    round_logs: List[Dict[str, Any]] = []
    profits_prev: Dict[str, float] = {firm.firm_id: 0.0 for firm in firms}

    for round_index in range(1, rounds + 1):
        if pending_notices:
            visible_notices.extend(pending_notices)
            pending_notices = []
        if pending_replies:
            visible_replies.extend(pending_replies)
            pending_replies = []

        completed_projects = _complete_internal_projects_for_round(
            current_round=round_index,
            internal_projects_by_firm=internal_projects_by_firm,
            completed_internal_capabilities_by_firm=completed_internal_capabilities_by_firm,
        )
        total_internal_projects_completed += len(completed_projects)

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
                completed_internal_capabilities_by_firm=completed_internal_capabilities_by_firm,
                internal_projects_by_firm=internal_projects_by_firm,
                visible_notices=visible_notices,
                visible_replies=visible_replies,
                reputation=reputation,
                last_round_public_outcomes=last_round_public_outcomes,
                history=history_by_firm[firm.firm_id],
                score=profits_prev.get(firm.firm_id, 0.0),
                rank=ranks.get(firm.firm_id, len(firms)),
                total_firms=len(firms),
                fulfilled_orders=fulfilled_orders,
                current_round=round_index,
                total_rounds=rounds,
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
            internal_projects, public_notices, reply_notices, fulfill_orders, commentary = parse_firm_plan(
                data
            )
            plans[firm.firm_id] = {
                "internal_projects": internal_projects,
                "public_notices": public_notices,
                "reply_notices": reply_notices,
                "fulfill_orders": fulfill_orders,
                "commentary": commentary,
                "raw_text": raw_text,
            }
            if metadata.get("usage"):
                usage_by_firm[firm.firm_id] = metadata["usage"]

        notice_post_firm_rounds += sum(1 for plan in plans.values() if plan["public_notices"])
        reply_firm_rounds += sum(1 for plan in plans.values() if plan["reply_notices"])
        board_use_firm_rounds += sum(
            1
            for plan in plans.values()
            if plan["public_notices"] or plan["reply_notices"]
        )
        total_notices_posted += sum(len(plan["public_notices"]) for plan in plans.values())
        total_replies_posted += sum(len(plan["reply_notices"]) for plan in plans.values())

        internal_projects_started: List[Dict[str, Any]] = []
        for firm in firms:
            firm_id = firm.firm_id
            available_capabilities = _available_capabilities_for_firm(
                firm_id=firm_id,
                holdings_by_firm=holdings_by_firm,
                completed_internal_capabilities_by_firm=completed_internal_capabilities_by_firm,
                modules_by_id=modules_by_id,
            )
            active_capabilities = {
                project.capability for project in internal_projects_by_firm.get(firm_id, [])
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
                if module.capability in available_capabilities:
                    continue
                if module.capability in active_capabilities:
                    continue

                cost = module.substitute_costs[firm_id]
                project = InternalProject(
                    firm_id=firm_id,
                    capability=module.capability,
                    source_module_id=module.tech_id,
                    start_round=round_index,
                    ready_round=round_index + 2,
                    cost=cost,
                )
                internal_projects_by_firm[firm_id].append(project)
                active_capabilities.add(module.capability)
                round_profits[firm_id] -= cost
                total_welfare -= cost
                total_internal_projects_started += 1
                touched_orders = _orders_touched_by_capability(
                    firm_id=firm_id,
                    capability=module.capability,
                    orders_by_firm=orders_by_firm,
                    fulfilled_orders=fulfilled_orders,
                )
                started_orders.update(touched_orders)
                internal_projects_started.append(
                    {
                        "firm_id": firm_id,
                        "objective": objective,
                        "capability": module.capability,
                        "project_brief": module.order_phrase,
                        "source_module_id": module.tech_id,
                        "cost": round(cost, 4),
                        "ready_round": round_index + 2,
                        "touched_orders": touched_orders,
                    }
                )

        posted_notice_objects: List[BoardNotice] = []
        posted_notices: List[Dict[str, Any]] = []
        for firm in firms:
            for notice_data in plans[firm.firm_id]["public_notices"]:
                notice = BoardNotice(
                    notice_id=f"N{notice_counter}",
                    author=firm.firm_id,
                    headline=notice_data["headline"],
                    body=notice_data["body"],
                    cash_terms=notice_data["cash_terms"],
                    round_posted=round_index,
                )
                notice_counter += 1
                pending_notices.append(notice)
                posted_notice_objects.append(notice)
                posted_notices.append(_board_notice_to_dict(notice))

        posted_reply_objects: List[BoardReply] = []
        posted_replies: List[Dict[str, Any]] = []
        for firm in firms:
            for reply_data in plans[firm.firm_id]["reply_notices"]:
                reply = BoardReply(
                    reply_id=f"R{reply_counter}",
                    notice_id=reply_data["notice_id"],
                    author=firm.firm_id,
                    body=reply_data["body"],
                    cash_offer=float(reply_data["cash_offer"]),
                    round_posted=round_index,
                )
                reply_counter += 1
                pending_replies.append(reply)
                posted_reply_objects.append(reply)
                posted_replies.append(_board_reply_to_dict(reply))

        visible_notice_lookup = {notice.notice_id: notice for notice in visible_notices}
        successful_deals: List[Dict[str, Any]] = []
        failed_deals: List[Dict[str, Any]] = []
        unresolved_replies: List[Dict[str, Any]] = []
        seen_replies: Set[Tuple[str, str]] = set()

        for reply in posted_reply_objects:
            dedupe_key = (reply.author, reply.notice_id)
            if dedupe_key in seen_replies:
                continue
            seen_replies.add(dedupe_key)

            notice = visible_notice_lookup.get(reply.notice_id)
            if notice is None:
                unresolved_replies.append(
                    {
                        "reply_id": reply.reply_id,
                        "notice_id": reply.notice_id,
                        "buyer": reply.author,
                        "reason": "notice_not_visible",
                    }
                )
                continue
            if notice.author == reply.author:
                unresolved_replies.append(
                    {
                        "reply_id": reply.reply_id,
                        "notice_id": reply.notice_id,
                        "buyer": reply.author,
                        "reason": "self_reply",
                    }
                )
                continue
            if not _notice_is_offer_like(notice):
                unresolved_replies.append(
                    {
                        "reply_id": reply.reply_id,
                        "notice_id": reply.notice_id,
                        "buyer": reply.author,
                        "seller": notice.author,
                        "reason": "notice_not_offer_like",
                    }
                )
                continue

            seller_modules = modules_by_owner.get(notice.author, [])
            module = _resolve_module_from_text(
                text=_notice_text(notice),
                candidate_modules=seller_modules,
                min_score=2,
            )
            if module is None:
                unresolved_replies.append(
                    {
                        "reply_id": reply.reply_id,
                        "notice_id": reply.notice_id,
                        "buyer": reply.author,
                        "seller": notice.author,
                        "reason": "ambiguous_notice",
                    }
                )
                continue

            if module.tech_id in holdings_by_firm.get(reply.author, set()):
                unresolved_replies.append(
                    {
                        "reply_id": reply.reply_id,
                        "notice_id": reply.notice_id,
                        "buyer": reply.author,
                        "seller": notice.author,
                        "capability": module.capability,
                        "reason": "already_controlled",
                    }
                )
                continue

            reservation_price = round(module.reference_license_price + 0.25, 4)
            if reply.cash_offer < reservation_price:
                unresolved_replies.append(
                    {
                        "reply_id": reply.reply_id,
                        "notice_id": reply.notice_id,
                        "buyer": reply.author,
                        "seller": notice.author,
                        "capability": module.capability,
                        "cash_offer": round(reply.cash_offer, 4),
                        "reservation_price": reservation_price,
                        "reason": "offer_too_low",
                    }
                )
                continue

            clearing_price = round((reply.cash_offer + reservation_price) / 2.0, 4)
            if _deal_failure(
                seed=seed,
                round_index=round_index,
                tech_id=module.tech_id,
                seller_id=notice.author,
                buyer_id=reply.author,
                quality=module.quality,
            ):
                reputation[notice.author] -= 2.0
                reputation[reply.author] -= 1.0
                failed_deals.append(
                    {
                        "reply_id": reply.reply_id,
                        "notice_id": reply.notice_id,
                        "tech_id": module.tech_id,
                        "capability": module.capability,
                        "headline": notice.headline,
                        "seller": notice.author,
                        "buyer": reply.author,
                        "price": clearing_price,
                        "reason": "quality_failure",
                    }
                )
                continue

            holdings_by_firm.setdefault(reply.author, set()).add(module.tech_id)
            brokered_modules_by_firm[reply.author].add(module.tech_id)
            reputation[notice.author] += 1.0
            reputation[reply.author] += 0.5

            buyer_profit = -clearing_price - module.integration_costs[reply.author]
            seller_profit = clearing_price
            round_profits[reply.author] += buyer_profit
            round_profits[notice.author] += seller_profit
            total_welfare -= module.integration_costs[reply.author]
            total_deal_volume += 1
            touched_orders = _orders_touched_by_capability(
                firm_id=reply.author,
                capability=module.capability,
                orders_by_firm=orders_by_firm,
                fulfilled_orders=fulfilled_orders,
            )
            started_orders.update(touched_orders)
            successful_deals.append(
                {
                    "reply_id": reply.reply_id,
                    "notice_id": reply.notice_id,
                    "tech_id": module.tech_id,
                    "capability": module.capability,
                    "headline": notice.headline,
                    "seller": notice.author,
                    "buyer": reply.author,
                    "price": clearing_price,
                    "buyer_profit": round(buyer_profit, 4),
                    "seller_profit": round(seller_profit, 4),
                    "reply_body": reply.body,
                    "touched_orders": touched_orders,
                }
            )

        available_capabilities_after_trade = _available_capabilities_by_firm(
            firms=firms,
            holdings_by_firm=holdings_by_firm,
            completed_internal_capabilities_by_firm=completed_internal_capabilities_by_firm,
            modules_by_id=modules_by_id,
        )

        fulfillment_attempts_by_firm = {
            firm_id: list(plan["fulfill_orders"])
            for firm_id, plan in plans.items()
            if plan["fulfill_orders"]
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
            internal_projects_by_firm=internal_projects_by_firm,
        )
        started_not_delivered = started_not_delivered_orders(order_state_after)

        for firm in firms:
            firm_id = firm.firm_id
            history_by_firm[firm_id].append(
                {
                    "round": round_index,
                    "profit": round(round_profits[firm_id], 4),
                    "reputation": round(reputation[firm_id], 4),
                    "board_activity": {
                        "posted_notice": bool(plans[firm_id]["public_notices"]),
                        "posted_reply": bool(plans[firm_id]["reply_notices"]),
                    },
                    "successful_deals": [
                        {
                            "counterparty": (
                                deal["seller"] if deal["buyer"] == firm_id else deal["buyer"]
                            ),
                            "capability": deal["capability"],
                            "price": deal["price"],
                        }
                        for deal in successful_deals
                        if deal["buyer"] == firm_id or deal["seller"] == firm_id
                    ],
                    "successful_fulfillments": [
                        fulfillment
                        for fulfillment in successful_fulfillments
                        if fulfillment["firm_id"] == firm_id
                    ],
                }
            )

        round_usage = _round_usage(usage_by_firm)
        round_logs.append(
            {
                "round": round_index,
                "visible_board": {
                    "notices": [_board_notice_to_dict(notice) for notice in visible_notices],
                    "replies": [_board_reply_to_dict(reply) for reply in visible_replies],
                },
                "plans": {
                    firm_id: {
                        "internal_projects": plan["internal_projects"],
                        "public_notices": plan["public_notices"],
                        "reply_notices": plan["reply_notices"],
                        "fulfill_orders": plan["fulfill_orders"],
                        "commentary": plan["commentary"],
                    }
                    for firm_id, plan in plans.items()
                },
                "internal_projects_started": internal_projects_started,
                "internal_projects_completed": completed_projects,
                "posted_notices": posted_notices,
                "posted_replies": posted_replies,
                "successful_deals": successful_deals,
                "failed_deals": failed_deals,
                "unresolved_replies": unresolved_replies,
                "fulfillment_attempts": fulfillment_attempts_by_firm,
                "successful_fulfillments": successful_fulfillments,
                "failed_fulfillments": failed_fulfillments,
                "order_state": order_state_after,
                "started_not_delivered": started_not_delivered,
                "holdings_by_firm": {
                    firm_id: sorted(holdings) for firm_id, holdings in holdings_by_firm.items()
                },
                "completed_internal_capabilities_by_firm": {
                    firm_id: sorted(items)
                    for firm_id, items in completed_internal_capabilities_by_firm.items()
                },
                "active_internal_projects": {
                    firm_id: [
                        {
                            "capability": project.capability,
                            "ready_round": project.ready_round,
                            "cost": round(project.cost, 4),
                        }
                        for project in projects
                    ]
                    for firm_id, projects in internal_projects_by_firm.items()
                },
                "notice_post_firms": sorted(
                    firm_id
                    for firm_id, plan in plans.items()
                    if plan["public_notices"]
                ),
                "reply_firms": sorted(
                    firm_id
                    for firm_id, plan in plans.items()
                    if plan["reply_notices"]
                ),
                "board_use_firms": sorted(
                    firm_id
                    for firm_id, plan in plans.items()
                    if plan["public_notices"] or plan["reply_notices"]
                ),
                "reputation": {firm_id: round(value, 4) for firm_id, value in reputation.items()},
                "profits": {firm_id: round(value, 4) for firm_id, value in round_profits.items()},
                "welfare": round(total_welfare, 4),
                "customer_value_captured": round(customer_value_captured, 4),
                "codex_usage": round_usage,
            }
        )

        profits_prev = dict(round_profits)
        last_round_public_outcomes = _public_outcomes(
            successful_deals=successful_deals,
            failed_deals=failed_deals,
            successful_fulfillments=successful_fulfillments,
        )

    final_available_capabilities = _available_capabilities_by_firm(
        firms=firms,
        holdings_by_firm=holdings_by_firm,
        completed_internal_capabilities_by_firm=completed_internal_capabilities_by_firm,
        modules_by_id=modules_by_id,
    )
    final_order_state = order_state_snapshot(
        orders=orders,
        available_capabilities_by_firm=final_available_capabilities,
        started_orders=started_orders,
        fulfilled_orders=fulfilled_orders,
        internal_projects_by_firm=internal_projects_by_firm,
    )
    summary = build_summary(
        orders=orders,
        order_state=final_order_state,
        brokered_deal_volume=total_deal_volume,
        public_notices_posted=total_notices_posted,
        replies_posted=total_replies_posted,
        board_use_firm_rounds=board_use_firm_rounds,
        notice_post_firm_rounds=notice_post_firm_rounds,
        reply_firm_rounds=reply_firm_rounds,
        total_firm_rounds=len(firms) * rounds,
        internal_projects_started=total_internal_projects_started,
        internal_projects_completed=total_internal_projects_completed,
        total_welfare=total_welfare,
        customer_value_captured=customer_value_captured,
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
        "brokered_modules_by_firm": {
            firm_id: sorted(items) for firm_id, items in brokered_modules_by_firm.items()
        },
        "completed_internal_capabilities_by_firm": {
            firm_id: sorted(items)
            for firm_id, items in completed_internal_capabilities_by_firm.items()
        },
    }
