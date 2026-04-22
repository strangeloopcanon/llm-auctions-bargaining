from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Sequence, Set, Tuple

from ..agents import call_agent_json, openai_client_from_settings
from ..protocols import safe_float
from ..settings import ModelSettings
from .models import BoardNotice, BoardReply, CustomerOrder, Firm, InventoryLot


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
    "can deliver",
    "offering",
    "offer",
    "provide",
    "supply",
)

INSTITUTION_KEYWORDS = (
    "escrow",
    "inspection",
    "inspect",
    "verification",
    "verify",
    "quality check",
    "hold payment",
    "release payment",
)

SUPPORT_KEYWORDS = (
    "support",
    "agree",
    "use",
    "join",
    "adopt",
    "accept",
)


def parse_firm_plan(
    raw: Dict[str, Any],
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]], List[str], str]:
    if not isinstance(raw, dict):
        return [], [], [], ""

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
    return public_notices, reply_notices, fulfill_orders, commentary


def _tokenize(text: str) -> Set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text or "").lower())
        if len(token) >= 3 and token not in STOPWORDS
    }


def _lot_lookup_text(lot: InventoryLot) -> str:
    return " ".join([lot.capability, lot.lot_name, lot.description, lot.order_phrase])


def _lot_match_score(text: str, lot: InventoryLot) -> int:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return 0
    overlap = len(_tokenize(normalized) & _tokenize(_lot_lookup_text(lot)))
    if lot.capability.lower() in normalized:
        overlap += 3
    if lot.lot_name.lower() in normalized:
        overlap += 2
    if lot.order_phrase.lower() in normalized:
        overlap += 2
    return overlap


def _resolve_lot_from_text(
    *,
    text: str,
    candidate_lots: Sequence[InventoryLot],
    min_score: int,
) -> InventoryLot | None:
    best_lot: InventoryLot | None = None
    best_score = 0
    second_best = 0
    for lot in candidate_lots:
        score = _lot_match_score(text, lot)
        if score > best_score:
            second_best = best_score
            best_score = score
            best_lot = lot
            continue
        if score > second_best:
            second_best = score
    if best_lot is None or best_score < min_score:
        return None
    if best_score == second_best:
        return None
    return best_lot


def _notice_text(notice: BoardNotice) -> str:
    return " ".join([notice.headline, notice.body, notice.cash_terms]).strip()


def _notice_is_offer_like(notice: BoardNotice) -> bool:
    text = _notice_text(notice).lower()
    return any(keyword in text for keyword in OFFER_KEYWORDS)


def _notice_is_institution_proposal(notice: BoardNotice) -> bool:
    text = _notice_text(notice).lower()
    return any(keyword in text for keyword in INSTITUTION_KEYWORDS)


def _reply_supports_institution(reply: BoardReply) -> bool:
    text = reply.body.lower()
    return any(keyword in text for keyword in SUPPORT_KEYWORDS)


def _lot_index(lots: Sequence[InventoryLot]) -> Dict[str, InventoryLot]:
    return {lot.lot_id: lot for lot in lots}


def _lots_by_seller(lots: Sequence[InventoryLot]) -> Dict[str, List[InventoryLot]]:
    grouped: Dict[str, List[InventoryLot]] = {}
    for lot in lots:
        grouped.setdefault(lot.seller, []).append(lot)
    return grouped


def _order_index(orders: Sequence[CustomerOrder]) -> Dict[str, CustomerOrder]:
    return {order.order_id: order for order in orders}


def _orders_by_buyer(orders: Sequence[CustomerOrder]) -> Dict[str, List[CustomerOrder]]:
    grouped: Dict[str, List[CustomerOrder]] = {}
    for order in orders:
        grouped.setdefault(order.buyer, []).append(order)
    return grouped


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


def _public_reputation(firms: Sequence[Firm], reputation: Dict[str, float]) -> List[Dict[str, Any]]:
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
    orders_by_buyer: Dict[str, List[CustomerOrder]],
    acquired_lots_by_buyer: Dict[str, Set[str]],
    lots_by_seller: Dict[str, List[InventoryLot]],
    sold_lots: Set[str],
    lots_by_id: Dict[str, InventoryLot],
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
    customer_orders = []
    for order in orders_by_buyer.get(firm.firm_id, []):
        customer_orders.append(
            {
                "order_id": order.order_id,
                "customer_brief": order.customer_brief,
                "delivery_value": order.delivery_value,
                "deadline_round": order.deadline_round,
                "fulfilled": order.order_id in fulfilled_orders,
            }
        )

    acquired_inputs = []
    for lot_id in sorted(acquired_lots_by_buyer.get(firm.firm_id, set())):
        lot = lots_by_id.get(lot_id)
        if lot is None:
            continue
        acquired_inputs.append(
            {
                "lot_name": lot.lot_name,
                "capability": lot.capability,
                "description": lot.description,
            }
        )

    inventory_lots = []
    for lot in lots_by_seller.get(firm.firm_id, []):
        inventory_lots.append(
            {
                "lot_id": lot.lot_id,
                "lot_name": lot.lot_name,
                "description": lot.description,
                "order_phrase": lot.order_phrase,
                "reservation_price": lot.reservation_price,
                "quality": lot.quality,
                "already_sold": lot.lot_id in sold_lots,
            }
        )

    return {
        "firm_id": firm.firm_id,
        "role": firm.role,
        "current_round": current_round,
        "total_rounds": total_rounds,
        "customer_orders": customer_orders,
        "acquired_inputs": acquired_inputs,
        "inventory_lots": inventory_lots,
        "public_board": {
            "visible_notices": [_board_notice_to_dict(notice) for notice in visible_notices],
            "visible_replies": [_board_reply_to_dict(reply) for reply in visible_replies],
        },
        "public_reputation": _public_reputation(firms=sorted(firms, key=lambda item: item.firm_id), reputation=reputation),
        "last_round_public_outcomes": list(last_round_public_outcomes),
        "recent_history": history[-3:],
        "your_score_last_round": round(score, 4),
        "your_rank_last_round": rank,
        "total_firms": total_firms,
    }


def _trade_failure_probability(quality: str, protected: bool) -> float:
    if protected:
        return {"High": 0.02, "Medium": 0.05, "Low": 0.08}.get(quality, 0.05)
    return {"High": 0.45, "Medium": 0.7, "Low": 0.9}.get(quality, 0.7)


def _trade_fails(
    *,
    seed: int,
    round_index: int,
    lot_id: str,
    seller_id: str,
    buyer_id: str,
    quality: str,
    protected: bool,
) -> bool:
    draw = random.Random(
        f"{seed}:{round_index}:{lot_id}:{seller_id}:{buyer_id}:{protected}"
    ).random()
    return draw < _trade_failure_probability(quality, protected)


def _available_capabilities_by_buyer(
    acquired_lots_by_buyer: Dict[str, Set[str]],
    lots_by_id: Dict[str, InventoryLot],
) -> Dict[str, Set[str]]:
    available: Dict[str, Set[str]] = {}
    for buyer, lot_ids in acquired_lots_by_buyer.items():
        available[buyer] = {
            lots_by_id[lot_id].capability for lot_id in lot_ids if lot_id in lots_by_id
        }
    return available


def order_state_snapshot(
    *,
    orders: Sequence[CustomerOrder],
    available_capabilities_by_buyer: Dict[str, Set[str]],
    started_orders: Set[str],
    fulfilled_orders: Set[str],
) -> Dict[str, Dict[str, Any]]:
    snapshot: Dict[str, Dict[str, Any]] = {}
    for order in orders:
        available = available_capabilities_by_buyer.get(order.buyer, set())
        covered = order.required_capability in available
        snapshot[order.order_id] = {
            "buyer": order.buyer,
            "required_capability": order.required_capability,
            "covered_capability": covered,
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
    available_capabilities_by_buyer: Dict[str, Set[str]],
    fulfillment_attempts_by_buyer: Dict[str, List[str]],
    fulfilled_orders: Set[str],
    current_round: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
    successful: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    customer_value = 0.0

    for buyer, order_ids in fulfillment_attempts_by_buyer.items():
        for order_id in order_ids:
            order = orders_by_id.get(order_id)
            if order is None:
                failed.append({"buyer": buyer, "order_id": order_id, "reason": "unknown_order"})
                continue
            if order.buyer != buyer:
                failed.append(
                    {
                        "buyer": buyer,
                        "order_id": order_id,
                        "reason": "wrong_buyer",
                        "target_buyer": order.buyer,
                    }
                )
                continue
            if order_id in fulfilled_orders:
                failed.append({"buyer": buyer, "order_id": order_id, "reason": "already_fulfilled"})
                continue
            if current_round > order.deadline_round:
                failed.append({"buyer": buyer, "order_id": order_id, "reason": "past_deadline"})
                continue
            available = available_capabilities_by_buyer.get(buyer, set())
            if order.required_capability not in available:
                failed.append(
                    {
                        "buyer": buyer,
                        "order_id": order_id,
                        "reason": "missing_capability",
                        "required_capability": order.required_capability,
                    }
                )
                continue
            fulfilled_orders.add(order_id)
            customer_value += order.delivery_value
            successful.append(
                {
                    "buyer": buyer,
                    "order_id": order_id,
                    "delivery_value": order.delivery_value,
                }
            )

    return successful, failed, customer_value


def _public_outcomes(
    *,
    active_institutions: Sequence[Dict[str, Any]],
    successful_deals: Sequence[Dict[str, Any]],
    failed_deals: Sequence[Dict[str, Any]],
    successful_fulfillments: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    outcomes: List[Dict[str, Any]] = []
    for institution in active_institutions:
        outcomes.append(
            {
                "kind": "institution_active",
                "notice_id": institution["notice_id"],
                "founder": institution["founder"],
                "activated_round": institution["activated_round"],
            }
        )
    for deal in successful_deals:
        outcomes.append(
            {
                "kind": "deal",
                "seller": deal["seller"],
                "buyer": deal["buyer"],
                "lot_name": deal["lot_name"],
                "price": deal["price"],
                "protected": deal["protected"],
            }
        )
    for deal in failed_deals:
        outcomes.append(
            {
                "kind": "failed_deal",
                "seller": deal["seller"],
                "buyer": deal["buyer"],
                "lot_name": deal["lot_name"],
                "reason": deal["reason"],
                "protected": deal["protected"],
            }
        )
    for fulfillment in successful_fulfillments:
        outcomes.append(
            {
                "kind": "delivery",
                "buyer": fulfillment["buyer"],
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
    institutions_activated: int,
    institution_post_firm_rounds: int,
    board_use_firm_rounds: int,
    reply_firm_rounds: int,
    total_firm_rounds: int,
    successful_deal_volume: int,
    safe_deal_volume: int,
    failed_deal_volume: int,
    total_welfare: float,
    customer_value_captured: float,
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
        "institution_notice_rate": round(
            institution_post_firm_rounds / total_firm_rounds, 4
        ) if total_firm_rounds else 0.0,
        "reply_rate": round(reply_firm_rounds / total_firm_rounds, 4) if total_firm_rounds else 0.0,
        "board_use_rate": round(
            board_use_firm_rounds / total_firm_rounds, 4
        ) if total_firm_rounds else 0.0,
        "institutions_activated": institutions_activated,
        "institution_activation_rate": 1.0 if institutions_activated else 0.0,
        "deal_volume": successful_deal_volume,
        "safe_deal_volume": safe_deal_volume,
        "failed_deal_volume": failed_deal_volume,
        "welfare": round(total_welfare, 4),
        "customer_value_captured": round(customer_value_captured, 4),
    }


def run_escrow_market(
    model_settings: ModelSettings,
    firms: Sequence[Firm],
    lots: Sequence[InventoryLot],
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

    lots_by_id = _lot_index(lots)
    lots_by_seller = _lots_by_seller(lots)
    orders_by_id = _order_index(orders)
    orders_by_buyer = _orders_by_buyer(orders)
    acquired_lots_by_buyer: Dict[str, Set[str]] = {firm.firm_id: set() for firm in firms}
    sold_lots: Set[str] = set()
    reputation = {firm.firm_id: 0.0 for firm in firms}
    history_by_firm: Dict[str, List[Dict[str, Any]]] = {firm.firm_id: [] for firm in firms}
    visible_notices: List[BoardNotice] = []
    visible_replies: List[BoardReply] = []
    pending_notices: List[BoardNotice] = []
    pending_replies: List[BoardReply] = []
    active_institutions: List[Dict[str, Any]] = []
    active_institution_notice_ids: Set[str] = set()
    last_round_public_outcomes: List[Dict[str, Any]] = []
    fulfilled_orders: Set[str] = set()
    started_orders: Set[str] = set()
    notice_counter = 1
    reply_counter = 1
    customer_value_captured = 0.0
    total_welfare = 0.0
    successful_deal_volume = 0
    safe_deal_volume = 0
    failed_deal_volume = 0
    institution_post_firm_rounds = 0
    board_use_firm_rounds = 0
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
                orders_by_buyer=orders_by_buyer,
                acquired_lots_by_buyer=acquired_lots_by_buyer,
                lots_by_seller=lots_by_seller,
                sold_lots=sold_lots,
                lots_by_id=lots_by_id,
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
            public_notices, reply_notices, fulfill_orders, commentary = parse_firm_plan(data)
            plans[firm.firm_id] = {
                "public_notices": public_notices,
                "reply_notices": reply_notices,
                "fulfill_orders": fulfill_orders,
                "commentary": commentary,
                "raw_text": raw_text,
            }
            if metadata.get("usage"):
                usage_by_firm[firm.firm_id] = metadata["usage"]

        institution_post_firm_rounds += sum(
            1
            for plan in plans.values()
            if any(
                _notice_is_institution_proposal(
                    BoardNotice("", "", notice["headline"], notice["body"], notice["cash_terms"], 0)
                )
                for notice in plan["public_notices"]
            )
        )
        reply_firm_rounds += sum(1 for plan in plans.values() if plan["reply_notices"])
        board_use_firm_rounds += sum(
            1
            for plan in plans.values()
            if plan["public_notices"] or plan["reply_notices"]
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
        institutions_activated_this_round: List[Dict[str, Any]] = []
        successful_deals: List[Dict[str, Any]] = []
        failed_deals: List[Dict[str, Any]] = []
        unresolved_replies: List[Dict[str, Any]] = []

        institution_replies: List[BoardReply] = []
        trade_replies: List[BoardReply] = []
        for reply in posted_reply_objects:
            notice = visible_notice_lookup.get(reply.notice_id)
            if notice is None:
                unresolved_replies.append(
                    {
                        "reply_id": reply.reply_id,
                        "notice_id": reply.notice_id,
                        "author": reply.author,
                        "reason": "notice_not_visible",
                    }
                )
                continue
            if _notice_is_institution_proposal(notice):
                institution_replies.append(reply)
                continue
            trade_replies.append(reply)

        for reply in institution_replies:
            notice = visible_notice_lookup[reply.notice_id]
            if notice.author == reply.author:
                unresolved_replies.append(
                    {
                        "reply_id": reply.reply_id,
                        "notice_id": reply.notice_id,
                        "author": reply.author,
                        "reason": "self_reply",
                    }
                )
                continue
            if notice.notice_id in active_institution_notice_ids:
                continue
            if not _reply_supports_institution(reply):
                unresolved_replies.append(
                    {
                        "reply_id": reply.reply_id,
                        "notice_id": reply.notice_id,
                        "author": reply.author,
                        "reason": "institution_reply_without_support",
                    }
                )
                continue
            active_institution_notice_ids.add(notice.notice_id)
            institution = {
                "notice_id": notice.notice_id,
                "founder": notice.author,
                "supporter": reply.author,
                "activated_round": round_index,
            }
            active_institutions.append(institution)
            institutions_activated_this_round.append(institution)
            reputation[notice.author] += 0.5
            reputation[reply.author] += 0.5

        for reply in trade_replies:
            notice = visible_notice_lookup[reply.notice_id]
            if _notice_is_institution_proposal(notice):
                continue
            if notice.author == reply.author:
                unresolved_replies.append(
                    {
                        "reply_id": reply.reply_id,
                        "notice_id": reply.notice_id,
                        "author": reply.author,
                        "reason": "self_reply",
                    }
                )
                continue
            if not _notice_is_offer_like(notice):
                unresolved_replies.append(
                    {
                        "reply_id": reply.reply_id,
                        "notice_id": reply.notice_id,
                        "author": reply.author,
                        "seller": notice.author,
                        "reason": "notice_not_offer_like",
                    }
                )
                continue

            candidate_lots = [
                lot for lot in lots_by_seller.get(notice.author, []) if lot.lot_id not in sold_lots
            ]
            lot = _resolve_lot_from_text(
                text=_notice_text(notice),
                candidate_lots=candidate_lots,
                min_score=2,
            )
            if lot is None:
                unresolved_replies.append(
                    {
                        "reply_id": reply.reply_id,
                        "notice_id": reply.notice_id,
                        "author": reply.author,
                        "seller": notice.author,
                        "reason": "ambiguous_notice",
                    }
                )
                continue
            if lot.lot_id in sold_lots:
                unresolved_replies.append(
                    {
                        "reply_id": reply.reply_id,
                        "notice_id": reply.notice_id,
                        "author": reply.author,
                        "seller": notice.author,
                        "lot_id": lot.lot_id,
                        "reason": "sold_out",
                    }
                )
                continue
            if reply.cash_offer < lot.reservation_price:
                unresolved_replies.append(
                    {
                        "reply_id": reply.reply_id,
                        "notice_id": reply.notice_id,
                        "author": reply.author,
                        "seller": notice.author,
                        "lot_id": lot.lot_id,
                        "cash_offer": round(reply.cash_offer, 4),
                        "reservation_price": round(lot.reservation_price, 4),
                        "reason": "offer_too_low",
                    }
                )
                continue

            protected = bool(active_institution_notice_ids)
            clearing_price = round((reply.cash_offer + lot.reservation_price) / 2.0, 4)
            if _trade_fails(
                seed=seed,
                round_index=round_index,
                lot_id=lot.lot_id,
                seller_id=notice.author,
                buyer_id=reply.author,
                quality=lot.quality,
                protected=protected,
            ):
                reputation[notice.author] -= 2.0
                reputation[reply.author] -= 1.0
                failed_deal_volume += 1
                failed_deals.append(
                    {
                        "reply_id": reply.reply_id,
                        "notice_id": reply.notice_id,
                        "lot_id": lot.lot_id,
                        "lot_name": lot.lot_name,
                        "seller": notice.author,
                        "buyer": reply.author,
                        "price": clearing_price,
                        "protected": protected,
                        "reason": "quality_failure",
                    }
                )
                continue

            sold_lots.add(lot.lot_id)
            acquired_lots_by_buyer.setdefault(reply.author, set()).add(lot.lot_id)
            buyer_profit = -clearing_price
            seller_profit = clearing_price
            round_profits[reply.author] += buyer_profit
            round_profits[notice.author] += seller_profit
            successful_deal_volume += 1
            if protected:
                safe_deal_volume += 1
            reputation[notice.author] += 1.0
            reputation[reply.author] += 0.5
            for order in orders_by_buyer.get(reply.author, []):
                if order.required_capability == lot.capability and order.order_id not in fulfilled_orders:
                    started_orders.add(order.order_id)
            successful_deals.append(
                {
                    "reply_id": reply.reply_id,
                    "notice_id": reply.notice_id,
                    "lot_id": lot.lot_id,
                    "lot_name": lot.lot_name,
                    "capability": lot.capability,
                    "seller": notice.author,
                    "buyer": reply.author,
                    "price": clearing_price,
                    "protected": protected,
                }
            )

        available_capabilities = _available_capabilities_by_buyer(
            acquired_lots_by_buyer=acquired_lots_by_buyer,
            lots_by_id=lots_by_id,
        )
        fulfillment_attempts_by_buyer = {
            firm_id: list(plan["fulfill_orders"])
            for firm_id, plan in plans.items()
            if plan["fulfill_orders"]
        }
        for order_ids in fulfillment_attempts_by_buyer.values():
            started_orders.update(order_ids)
        successful_fulfillments, failed_fulfillments, round_customer_value = evaluate_fulfillment_attempts(
            orders_by_id=orders_by_id,
            available_capabilities_by_buyer=available_capabilities,
            fulfillment_attempts_by_buyer=fulfillment_attempts_by_buyer,
            fulfilled_orders=fulfilled_orders,
            current_round=round_index,
        )
        for fulfillment in successful_fulfillments:
            round_profits[fulfillment["buyer"]] += fulfillment["delivery_value"]
        total_welfare += round_customer_value
        customer_value_captured += round_customer_value

        order_state_after = order_state_snapshot(
            orders=orders,
            available_capabilities_by_buyer=available_capabilities,
            started_orders=started_orders,
            fulfilled_orders=fulfilled_orders,
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
                            "lot_name": deal["lot_name"],
                            "price": deal["price"],
                            "protected": deal["protected"],
                        }
                        for deal in successful_deals
                        if deal["buyer"] == firm_id or deal["seller"] == firm_id
                    ],
                }
            )

        round_logs.append(
            {
                "round": round_index,
                "visible_board": {
                    "notices": [_board_notice_to_dict(notice) for notice in visible_notices],
                    "replies": [_board_reply_to_dict(reply) for reply in visible_replies],
                },
                "plans": {
                    firm_id: {
                        "public_notices": plan["public_notices"],
                        "reply_notices": plan["reply_notices"],
                        "fulfill_orders": plan["fulfill_orders"],
                        "commentary": plan["commentary"],
                    }
                    for firm_id, plan in plans.items()
                },
                "posted_notices": posted_notices,
                "posted_replies": posted_replies,
                "institutions_activated": institutions_activated_this_round,
                "active_institutions": list(active_institutions),
                "successful_deals": successful_deals,
                "failed_deals": failed_deals,
                "unresolved_replies": unresolved_replies,
                "fulfillment_attempts": fulfillment_attempts_by_buyer,
                "successful_fulfillments": successful_fulfillments,
                "failed_fulfillments": failed_fulfillments,
                "order_state": order_state_after,
                "started_not_delivered": started_not_delivered,
                "sold_lots": sorted(sold_lots),
                "acquired_lots_by_buyer": {
                    buyer: sorted(lot_ids) for buyer, lot_ids in acquired_lots_by_buyer.items()
                },
                "reputation": {firm_id: round(value, 4) for firm_id, value in reputation.items()},
                "profits": {firm_id: round(value, 4) for firm_id, value in round_profits.items()},
                "welfare": round(total_welfare, 4),
                "customer_value_captured": round(customer_value_captured, 4),
                "codex_usage": _round_usage(usage_by_firm),
            }
        )

        profits_prev = dict(round_profits)
        last_round_public_outcomes = _public_outcomes(
            active_institutions=institutions_activated_this_round,
            successful_deals=successful_deals,
            failed_deals=failed_deals,
            successful_fulfillments=successful_fulfillments,
        )

    final_available_capabilities = _available_capabilities_by_buyer(
        acquired_lots_by_buyer=acquired_lots_by_buyer,
        lots_by_id=lots_by_id,
    )
    final_order_state = order_state_snapshot(
        orders=orders,
        available_capabilities_by_buyer=final_available_capabilities,
        started_orders=started_orders,
        fulfilled_orders=fulfilled_orders,
    )
    summary = build_summary(
        orders=orders,
        order_state=final_order_state,
        institutions_activated=len(active_institutions),
        institution_post_firm_rounds=institution_post_firm_rounds,
        board_use_firm_rounds=board_use_firm_rounds,
        reply_firm_rounds=reply_firm_rounds,
        total_firm_rounds=len(firms) * rounds,
        successful_deal_volume=successful_deal_volume,
        safe_deal_volume=safe_deal_volume,
        failed_deal_volume=failed_deal_volume,
        total_welfare=total_welfare,
        customer_value_captured=customer_value_captured,
    )

    return {
        "summary": summary,
        "round_logs": round_logs,
        "orders": [
            {
                "order_id": order.order_id,
                "buyer": order.buyer,
                "customer_brief": order.customer_brief,
                "delivery_value": order.delivery_value,
                "deadline_round": order.deadline_round,
                "required_capability": order.required_capability,
            }
            for order in orders
        ],
        "lots": [
            {
                "lot_id": lot.lot_id,
                "seller": lot.seller,
                "capability": lot.capability,
                "lot_name": lot.lot_name,
                "quality": lot.quality,
                "reservation_price": lot.reservation_price,
            }
            for lot in lots
        ],
    }
