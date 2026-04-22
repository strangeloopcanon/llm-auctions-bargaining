from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence, Tuple

from ..agents import call_agent_json, openai_client_from_settings
from ..settings import ModelSettings
from .models import BoardNotice, BoardReply, CustomerOrder, Firm, SlotClaim


SCHEDULE_KEYWORDS = (
    "booking",
    "calendar",
    "queue",
    "reserve",
    "schedule",
    "slot",
)

SHIPPING_KEYWORDS = (
    "dispatch",
    "dock",
    "pickup",
    "ship",
    "shipping",
    "truck",
)

CERTIFICATION_KEYWORDS = (
    "certification",
    "certify",
    "lab",
    "test",
    "testing",
)


def parse_firm_plan(
    raw: Dict[str, Any],
) -> Tuple[List[str], List[Dict[str, str]], List[Dict[str, str]], List[str], List[str], str]:
    if not isinstance(raw, dict):
        return [], [], [], [], [], ""

    build_products = [
        str(item).strip()
        for item in list(raw.get("build_products") or [])
        if str(item).strip()
    ]

    public_notices: List[Dict[str, str]] = []
    for item in list(raw.get("public_notices") or []):
        if not isinstance(item, dict):
            continue
        headline = str(item.get("headline") or "").strip()
        body = str(item.get("body") or "").strip()
        if not headline and not body:
            continue
        public_notices.append({"headline": headline, "body": body})

    reply_notices: List[Dict[str, str]] = []
    for item in list(raw.get("reply_notices") or []):
        if not isinstance(item, dict):
            continue
        notice_id = str(item.get("notice_id") or "").strip()
        body = str(item.get("body") or "").strip()
        if not notice_id or not body:
            continue
        reply_notices.append({"notice_id": notice_id, "body": body})

    submit_for_certification = [
        str(item).strip()
        for item in list(raw.get("submit_for_certification") or [])
        if str(item).strip()
    ]
    fulfill_orders = [
        str(item).strip()
        for item in list(raw.get("fulfill_orders") or [])
        if str(item).strip()
    ]
    commentary = str(raw.get("commentary") or "").strip()
    return (
        build_products,
        public_notices,
        reply_notices,
        submit_for_certification,
        fulfill_orders,
        commentary,
    )


def _board_notice_to_dict(notice: BoardNotice) -> Dict[str, Any]:
    return {
        "notice_id": notice.notice_id,
        "author": notice.author,
        "headline": notice.headline,
        "body": notice.body,
        "round_posted": notice.round_posted,
    }


def _board_reply_to_dict(reply: BoardReply) -> Dict[str, Any]:
    return {
        "reply_id": reply.reply_id,
        "notice_id": reply.notice_id,
        "author": reply.author,
        "body": reply.body,
        "round_posted": reply.round_posted,
    }


def _slot_claim_to_dict(claim: SlotClaim) -> Dict[str, Any]:
    return {
        "claim_id": claim.claim_id,
        "resource": claim.resource,
        "notice_id": claim.notice_id,
        "claimant": claim.claimant,
        "order_id": claim.order_id,
        "target_round": claim.target_round,
        "round_posted": claim.round_posted,
        "accepted": claim.accepted,
        "used": claim.used,
    }


def _orders_by_firm(orders: Sequence[CustomerOrder]) -> Dict[str, List[CustomerOrder]]:
    grouped: Dict[str, List[CustomerOrder]] = {}
    for order in orders:
        grouped.setdefault(order.firm_id, []).append(order)
    return grouped


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _resolve_order_reference(
    *,
    reference: str,
    firm_id: str,
    orders_by_firm: Dict[str, List[CustomerOrder]],
) -> CustomerOrder | None:
    firm_orders = orders_by_firm.get(firm_id, [])
    if not firm_orders:
        return None
    if len(firm_orders) == 1:
        return firm_orders[0]

    normalized = _normalize_text(reference)
    for order in firm_orders:
        if normalized == _normalize_text(order.order_id):
            return order
        if normalized == _normalize_text(order.product_name):
            return order
    for order in firm_orders:
        if normalized and normalized in _normalize_text(order.customer_brief):
            return order
    return None


def _notice_text(notice: BoardNotice) -> str:
    return " ".join([notice.headline, notice.body]).strip().lower()


def _notice_resource(notice: BoardNotice) -> str | None:
    text = _notice_text(notice)
    if not any(keyword in text for keyword in SCHEDULE_KEYWORDS):
        return None
    if any(keyword in text for keyword in SHIPPING_KEYWORDS):
        return "shipping"
    if any(keyword in text for keyword in CERTIFICATION_KEYWORDS):
        return "certification"
    return "certification"


def _active_schedule_notice(
    visible_notices: Sequence[BoardNotice],
    *,
    resource: str,
) -> BoardNotice | None:
    candidates = [
        notice for notice in visible_notices if _notice_resource(notice) == resource
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item.round_posted, item.notice_id))
    return candidates[0]


def _extract_order_id(text: str) -> str:
    match = re.search(r"\b(O\d+)\b", str(text or ""), flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).upper()


def _extract_round_number(text: str) -> int | None:
    explicit_match = re.search(r"\bround\s*(\d+)\b", str(text or ""), flags=re.IGNORECASE)
    if explicit_match:
        return int(explicit_match.group(1))
    fallback_match = re.search(r"\b(\d+)\b", str(text or ""))
    if fallback_match:
        return int(fallback_match.group(1))
    return None


def _slot_ledger(
    *,
    claims: Sequence[SlotClaim],
    resource: str,
    total_rounds: int,
    capacity_per_round: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for round_number in range(1, total_rounds + 1):
        accepted_claims = [
            claim
            for claim in claims
            if claim.accepted
            and claim.resource == resource
            and claim.target_round == round_number
        ]
        accepted_claims.sort(key=lambda item: item.claim_id)
        rows.append(
            {
                "round": round_number,
                "capacity": capacity_per_round,
                "reserved": len(accepted_claims),
                "remaining_capacity": max(0, capacity_per_round - len(accepted_claims)),
                "claims": [_slot_claim_to_dict(claim) for claim in accepted_claims],
            }
        )
    return rows


def _open_claim_for_order(
    *,
    claims: Sequence[SlotClaim],
    resource: str,
    order_id: str,
    current_round: int,
) -> SlotClaim | None:
    candidates = [
        claim
        for claim in claims
        if claim.accepted
        and not claim.used
        and claim.resource == resource
        and claim.order_id == order_id
        and claim.target_round >= current_round
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item.target_round, item.claim_id))
    return candidates[0]


def _current_round_claim(
    *,
    claims: Sequence[SlotClaim],
    resource: str,
    firm_id: str,
    order_id: str,
    current_round: int,
) -> SlotClaim | None:
    for claim in claims:
        if not claim.accepted:
            continue
        if claim.used:
            continue
        if claim.resource != resource:
            continue
        if claim.claimant != firm_id:
            continue
        if claim.order_id != order_id:
            continue
        if claim.target_round != current_round:
            continue
        return claim
    return None


def _resource_state(
    *,
    claims: Sequence[SlotClaim],
    resource: str,
    active_notice: BoardNotice | None,
    total_rounds: int,
    capacity_per_round: int,
) -> Dict[str, Any]:
    label = "shared lab" if resource == "certification" else "shared dock"
    return {
        "capacity_per_round": capacity_per_round,
        "active_booking_notice_id": active_notice.notice_id if active_notice else None,
        "booking_policy": (
            f"The {label} can process only products tied to an accepted current-round "
            "slot claim under a visible public booking rule."
        ),
        "slot_ledger": _slot_ledger(
            claims=claims,
            resource=resource,
            total_rounds=total_rounds,
            capacity_per_round=capacity_per_round,
        ),
    }


def _firm_private_input(
    *,
    firm: Firm,
    orders_by_firm: Dict[str, List[CustomerOrder]],
    built_orders: set[str],
    certified_orders: set[str],
    fulfilled_orders: set[str],
    visible_notices: Sequence[BoardNotice],
    visible_replies: Sequence[BoardReply],
    active_certification_notice: BoardNotice | None,
    active_shipping_notice: BoardNotice | None,
    slot_claims: Sequence[SlotClaim],
    history: List[Dict[str, Any]],
    score: float,
    rank: int,
    total_firms: int,
    current_round: int,
    total_rounds: int,
    certification_capacity: int,
    shipping_capacity: int,
) -> Dict[str, Any]:
    customer_orders = []
    for order in orders_by_firm.get(firm.firm_id, []):
        next_cert_slot = _open_claim_for_order(
            claims=slot_claims,
            resource="certification",
            order_id=order.order_id,
            current_round=current_round,
        )
        next_ship_slot = _open_claim_for_order(
            claims=slot_claims,
            resource="shipping",
            order_id=order.order_id,
            current_round=current_round,
        )
        customer_orders.append(
            {
                "order_id": order.order_id,
                "product_name": order.product_name,
                "customer_brief": order.customer_brief,
                "delivery_value": order.delivery_value,
                "build_cost": order.build_cost,
                "deadline_round": order.deadline_round,
                "built": order.order_id in built_orders,
                "certified": order.order_id in certified_orders,
                "fulfilled": order.order_id in fulfilled_orders,
                "next_certification_slot_round": next_cert_slot.target_round if next_cert_slot else None,
                "next_shipping_slot_round": next_ship_slot.target_round if next_ship_slot else None,
            }
        )

    return {
        "firm_id": firm.firm_id,
        "current_round": current_round,
        "total_rounds": total_rounds,
        "customer_orders": customer_orders,
        "certification_lab": _resource_state(
            claims=slot_claims,
            resource="certification",
            active_notice=active_certification_notice,
            total_rounds=total_rounds,
            capacity_per_round=certification_capacity,
        ),
        "shipping_dock": _resource_state(
            claims=slot_claims,
            resource="shipping",
            active_notice=active_shipping_notice,
            total_rounds=total_rounds,
            capacity_per_round=shipping_capacity,
        ),
        "public_board": {
            "visible_notices": [_board_notice_to_dict(notice) for notice in visible_notices],
            "visible_replies": [_board_reply_to_dict(reply) for reply in visible_replies],
        },
        "recent_history": history[-3:],
        "your_score_last_round": round(score, 4),
        "your_rank_last_round": rank,
        "total_firms": total_firms,
    }


def _public_outcomes(
    *,
    activated_certification_notice: BoardNotice | None,
    activated_shipping_notice: BoardNotice | None,
    accepted_claims: Sequence[SlotClaim],
    rejected_claims: Sequence[Dict[str, Any]],
    certification_successes: Sequence[Dict[str, Any]],
    certification_failures: Sequence[Dict[str, Any]],
    successful_fulfillments: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    outcomes: List[Dict[str, Any]] = []
    if activated_certification_notice is not None:
        outcomes.append(
            {
                "kind": "certification_booking_active",
                "notice_id": activated_certification_notice.notice_id,
                "founder": activated_certification_notice.author,
            }
        )
    if activated_shipping_notice is not None:
        outcomes.append(
            {
                "kind": "shipping_booking_active",
                "notice_id": activated_shipping_notice.notice_id,
                "founder": activated_shipping_notice.author,
            }
        )
    for claim in accepted_claims:
        outcomes.append(
            {
                "kind": "slot_reserved",
                "resource": claim.resource,
                "claimant": claim.claimant,
                "order_id": claim.order_id,
                "target_round": claim.target_round,
            }
        )
    for rejection in rejected_claims:
        outcomes.append({"kind": "slot_rejected", **rejection})
    for success in certification_successes:
        outcomes.append({"kind": "certified", **success})
    for failure in certification_failures:
        outcomes.append({"kind": "certification_failed", **failure})
    for fulfillment in successful_fulfillments:
        outcomes.append({"kind": "delivery", **fulfillment})
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
    built_orders: set[str],
    certified_orders: set[str],
    fulfilled_orders: set[str],
    certification_schedule_activated: bool,
    shipping_schedule_activated: bool,
    certification_notice_firm_rounds: int,
    shipping_notice_firm_rounds: int,
    certification_claim_firm_rounds: int,
    shipping_claim_firm_rounds: int,
    total_firm_rounds: int,
    certification_submissions: int,
    certification_slots_used: int,
    shipping_slots_used: int,
    total_build_cost: float,
    customer_value_captured: float,
) -> Dict[str, Any]:
    total_orders = len(orders)
    ready_but_undelivered = sorted(
        order.order_id
        for order in orders
        if order.order_id in built_orders and order.order_id not in fulfilled_orders
    )
    certified_but_undelivered = sorted(
        order.order_id
        for order in orders
        if order.order_id in certified_orders and order.order_id not in fulfilled_orders
    )
    return {
        "total_orders": total_orders,
        "products_built": len(built_orders),
        "certification_submissions": certification_submissions,
        "certification_successes": len(certified_orders),
        "certification_schedule_notice_rate": round(
            certification_notice_firm_rounds / total_firm_rounds, 4
        ) if total_firm_rounds else 0.0,
        "shipping_schedule_notice_rate": round(
            shipping_notice_firm_rounds / total_firm_rounds, 4
        ) if total_firm_rounds else 0.0,
        "certification_schedule_activation_rate": 1.0 if certification_schedule_activated else 0.0,
        "shipping_schedule_activation_rate": 1.0 if shipping_schedule_activated else 0.0,
        "certification_slot_claim_rate": round(
            certification_claim_firm_rounds / total_firm_rounds, 4
        ) if total_firm_rounds else 0.0,
        "shipping_slot_claim_rate": round(
            shipping_claim_firm_rounds / total_firm_rounds, 4
        ) if total_firm_rounds else 0.0,
        "certification_slots_used": certification_slots_used,
        "shipping_slots_used": shipping_slots_used,
        "orders_fulfilled": len(fulfilled_orders),
        "fulfillment_rate": round(
            len(fulfilled_orders) / total_orders, 4
        ) if total_orders else 0.0,
        "certified_but_undelivered_count": len(certified_but_undelivered),
        "certified_but_undelivered_orders": certified_but_undelivered,
        "ready_but_undelivered_count": len(ready_but_undelivered),
        "ready_but_undelivered_orders": ready_but_undelivered,
        "welfare": round(customer_value_captured - total_build_cost, 4),
        "customer_value_captured": round(customer_value_captured, 4),
    }


def _claim_resource_for_notice(
    *,
    notice_id: str,
    active_certification_notice: BoardNotice | None,
    active_shipping_notice: BoardNotice | None,
) -> str | None:
    if active_certification_notice is not None and notice_id == active_certification_notice.notice_id:
        return "certification"
    if active_shipping_notice is not None and notice_id == active_shipping_notice.notice_id:
        return "shipping"
    return None


def run_certification_market(
    model_settings: ModelSettings,
    firms: Sequence[Firm],
    orders: Sequence[CustomerOrder],
    *,
    dry_run: bool = False,
    rounds: int = 4,
    certification_capacity: int = 2,
    shipping_capacity: int | None = None,
) -> Dict[str, Any]:
    if shipping_capacity is None:
        shipping_capacity = certification_capacity

    client = None
    if not dry_run and model_settings.provider == "openai":
        client = openai_client_from_settings(model_settings)

    schema = {
        "type": "OBJECT",
        "properties": {
            "build_products": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
            "public_notices": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "headline": {"type": "STRING"},
                        "body": {"type": "STRING"},
                    },
                    "required": ["headline", "body"],
                },
            },
            "reply_notices": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "notice_id": {"type": "STRING"},
                        "body": {"type": "STRING"},
                    },
                    "required": ["notice_id", "body"],
                },
            },
            "submit_for_certification": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
            "fulfill_orders": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
            "commentary": {"type": "STRING"},
        },
    }

    orders_by_firm = _orders_by_firm(orders)
    built_orders: set[str] = set()
    certified_orders: set[str] = set()
    fulfilled_orders: set[str] = set()
    scores = {firm.firm_id: 0.0 for firm in firms}
    history_by_firm: Dict[str, List[Dict[str, Any]]] = {firm.firm_id: [] for firm in firms}
    visible_notices: List[BoardNotice] = []
    visible_replies: List[BoardReply] = []
    pending_notices: List[BoardNotice] = []
    pending_replies: List[BoardReply] = []
    slot_claims: List[SlotClaim] = []
    round_logs: List[Dict[str, Any]] = []
    total_build_cost = 0.0
    customer_value_captured = 0.0
    certification_submissions = 0
    certification_slots_used = 0
    shipping_slots_used = 0
    certification_notice_firm_rounds = 0
    shipping_notice_firm_rounds = 0
    certification_claim_firm_rounds = 0
    shipping_claim_firm_rounds = 0
    notice_counter = 1
    reply_counter = 1
    claim_counter = 1
    activated_certification_notice_id = ""
    activated_shipping_notice_id = ""
    certification_schedule_activated = False
    shipping_schedule_activated = False

    for current_round in range(1, rounds + 1):
        visible_notices = [*visible_notices, *pending_notices]
        visible_replies = [*visible_replies, *pending_replies]
        pending_notices = []
        pending_replies = []

        active_certification_notice = _active_schedule_notice(
            visible_notices,
            resource="certification",
        )
        active_shipping_notice = _active_schedule_notice(
            visible_notices,
            resource="shipping",
        )

        activated_certification_notice: BoardNotice | None = None
        activated_shipping_notice: BoardNotice | None = None

        if (
            active_certification_notice is not None
            and active_certification_notice.notice_id != activated_certification_notice_id
        ):
            activated_certification_notice_id = active_certification_notice.notice_id
            activated_certification_notice = active_certification_notice
            certification_schedule_activated = True

        if (
            active_shipping_notice is not None
            and active_shipping_notice.notice_id != activated_shipping_notice_id
        ):
            activated_shipping_notice_id = active_shipping_notice.notice_id
            activated_shipping_notice = active_shipping_notice
            shipping_schedule_activated = True

        ranked_scores = sorted(
            ((firm_id, score) for firm_id, score in scores.items()),
            key=lambda item: (-item[1], item[0]),
        )
        ranks = {firm_id: index + 1 for index, (firm_id, _) in enumerate(ranked_scores)}

        usage_by_firm: Dict[str, Dict[str, int]] = {}
        raw_plans_by_firm: Dict[str, Dict[str, Any]] = {}
        build_attempts_by_firm: Dict[str, List[str]] = {}
        submit_attempts_by_firm: Dict[str, List[str]] = {}
        fulfillment_attempts_by_firm: Dict[str, List[str]] = {}
        accepted_claims_this_round: List[SlotClaim] = []
        rejected_claims_this_round: List[Dict[str, Any]] = []

        for firm in sorted(firms, key=lambda item: item.firm_id):
            metadata_sink: Dict[str, Any] = {}
            private_input = _firm_private_input(
                firm=firm,
                orders_by_firm=orders_by_firm,
                built_orders=built_orders,
                certified_orders=certified_orders,
                fulfilled_orders=fulfilled_orders,
                visible_notices=visible_notices,
                visible_replies=visible_replies,
                active_certification_notice=active_certification_notice,
                active_shipping_notice=active_shipping_notice,
                slot_claims=slot_claims,
                history=history_by_firm[firm.firm_id],
                score=scores[firm.firm_id],
                rank=ranks.get(firm.firm_id, len(firms)),
                total_firms=len(firms),
                current_round=current_round,
                total_rounds=rounds,
                certification_capacity=certification_capacity,
                shipping_capacity=shipping_capacity,
            )
            raw, raw_text = call_agent_json(
                client=client,
                model_settings=model_settings,
                system_prompt=firm.system_prompt,
                user_prompt=json.dumps(private_input),
                response_schema=schema,
                dry_run=dry_run,
                metadata_sink=metadata_sink,
            )
            usage = metadata_sink.get("usage") or {}
            usage_by_firm[firm.firm_id] = {
                "calls": 1,
                "input_tokens": int(usage.get("input_tokens", 0) or 0),
                "output_tokens": int(usage.get("output_tokens", 0) or 0),
            }
            (
                build_products,
                public_notices,
                reply_notices,
                submit_for_certification,
                fulfill_orders,
                commentary,
            ) = parse_firm_plan(raw)
            raw_plans_by_firm[firm.firm_id] = {
                "build_products": build_products,
                "public_notices": public_notices,
                "reply_notices": reply_notices,
                "submit_for_certification": submit_for_certification,
                "fulfill_orders": fulfill_orders,
                "commentary": commentary,
                "raw_text": raw_text,
            }
            build_attempts_by_firm[firm.firm_id] = build_products
            submit_attempts_by_firm[firm.firm_id] = submit_for_certification
            fulfillment_attempts_by_firm[firm.firm_id] = fulfill_orders

            posted_certification_notice = False
            posted_shipping_notice = False
            for item in public_notices:
                notice = BoardNotice(
                    notice_id=f"N{notice_counter}",
                    author=firm.firm_id,
                    headline=item["headline"],
                    body=item["body"],
                    round_posted=current_round,
                )
                notice_counter += 1
                pending_notices.append(notice)
                resource = _notice_resource(notice)
                if resource == "certification":
                    posted_certification_notice = True
                if resource == "shipping":
                    posted_shipping_notice = True
            if posted_certification_notice:
                certification_notice_firm_rounds += 1
            if posted_shipping_notice:
                shipping_notice_firm_rounds += 1

            claimed_certification_slot = False
            claimed_shipping_slot = False
            for item in reply_notices:
                reply = BoardReply(
                    reply_id=f"R{reply_counter}",
                    notice_id=item["notice_id"],
                    author=firm.firm_id,
                    body=item["body"],
                    round_posted=current_round,
                )
                reply_counter += 1
                pending_replies.append(reply)

                resource = _claim_resource_for_notice(
                    notice_id=reply.notice_id,
                    active_certification_notice=active_certification_notice,
                    active_shipping_notice=active_shipping_notice,
                )
                if resource is None:
                    continue

                order_id = _extract_order_id(reply.body)
                order = None
                if order_id:
                    order = _resolve_order_reference(
                        reference=order_id,
                        firm_id=firm.firm_id,
                        orders_by_firm=orders_by_firm,
                    )
                if order is None:
                    order = _resolve_order_reference(
                        reference=reply.body,
                        firm_id=firm.firm_id,
                        orders_by_firm=orders_by_firm,
                    )
                if order is None:
                    rejected_claims_this_round.append(
                        {
                            "resource": resource,
                            "claimant": firm.firm_id,
                            "notice_id": reply.notice_id,
                            "reason": "unknown_order",
                        }
                    )
                    continue

                target_round = _extract_round_number(reply.body) or current_round
                if target_round < current_round or target_round > rounds:
                    rejected_claims_this_round.append(
                        {
                            "resource": resource,
                            "claimant": firm.firm_id,
                            "notice_id": reply.notice_id,
                            "order_id": order.order_id,
                            "target_round": target_round,
                            "reason": "invalid_round",
                        }
                    )
                    continue
                if target_round > order.deadline_round:
                    rejected_claims_this_round.append(
                        {
                            "resource": resource,
                            "claimant": firm.firm_id,
                            "notice_id": reply.notice_id,
                            "order_id": order.order_id,
                            "target_round": target_round,
                            "reason": "past_deadline",
                        }
                    )
                    continue
                if _open_claim_for_order(
                    claims=slot_claims,
                    resource=resource,
                    order_id=order.order_id,
                    current_round=current_round,
                ) is not None:
                    rejected_claims_this_round.append(
                        {
                            "resource": resource,
                            "claimant": firm.firm_id,
                            "notice_id": reply.notice_id,
                            "order_id": order.order_id,
                            "target_round": target_round,
                            "reason": "already_reserved",
                        }
                    )
                    continue

                capacity = certification_capacity if resource == "certification" else shipping_capacity
                reserved_this_round = sum(
                    1
                    for claim in slot_claims
                    if claim.accepted
                    and claim.resource == resource
                    and claim.target_round == target_round
                )
                if reserved_this_round >= capacity:
                    rejected_claims_this_round.append(
                        {
                            "resource": resource,
                            "claimant": firm.firm_id,
                            "notice_id": reply.notice_id,
                            "order_id": order.order_id,
                            "target_round": target_round,
                            "reason": "round_full",
                        }
                    )
                    continue

                claim = SlotClaim(
                    claim_id=f"C{claim_counter}",
                    resource=resource,
                    notice_id=reply.notice_id,
                    claimant=firm.firm_id,
                    order_id=order.order_id,
                    target_round=target_round,
                    round_posted=current_round,
                    accepted=True,
                )
                claim_counter += 1
                slot_claims.append(claim)
                accepted_claims_this_round.append(claim)
                if resource == "certification":
                    claimed_certification_slot = True
                else:
                    claimed_shipping_slot = True

            if claimed_certification_slot:
                certification_claim_firm_rounds += 1
            if claimed_shipping_slot:
                shipping_claim_firm_rounds += 1

        build_events: List[Dict[str, Any]] = []
        build_failures: List[Dict[str, Any]] = []
        for firm in sorted(firms, key=lambda item: item.firm_id):
            for reference in build_attempts_by_firm.get(firm.firm_id, []):
                order = _resolve_order_reference(
                    reference=reference,
                    firm_id=firm.firm_id,
                    orders_by_firm=orders_by_firm,
                )
                if order is None:
                    build_failures.append(
                        {"firm_id": firm.firm_id, "order_id": reference, "reason": "unknown_order"}
                    )
                    continue
                if order.order_id in built_orders:
                    build_failures.append(
                        {
                            "firm_id": firm.firm_id,
                            "order_id": order.order_id,
                            "reason": "already_built",
                        }
                    )
                    continue
                built_orders.add(order.order_id)
                scores[firm.firm_id] -= order.build_cost
                total_build_cost += order.build_cost
                build_events.append(
                    {
                        "firm_id": firm.firm_id,
                        "order_id": order.order_id,
                        "build_cost": order.build_cost,
                    }
                )

        certification_successes: List[Dict[str, Any]] = []
        certification_failures: List[Dict[str, Any]] = []
        for firm in sorted(firms, key=lambda item: item.firm_id):
            for reference in submit_attempts_by_firm.get(firm.firm_id, []):
                certification_submissions += 1
                order = _resolve_order_reference(
                    reference=reference,
                    firm_id=firm.firm_id,
                    orders_by_firm=orders_by_firm,
                )
                if order is None:
                    certification_failures.append(
                        {"firm_id": firm.firm_id, "order_id": reference, "reason": "unknown_order"}
                    )
                    continue
                if order.order_id not in built_orders:
                    certification_failures.append(
                        {"firm_id": firm.firm_id, "order_id": order.order_id, "reason": "not_built"}
                    )
                    continue
                if order.order_id in certified_orders:
                    certification_failures.append(
                        {
                            "firm_id": firm.firm_id,
                            "order_id": order.order_id,
                            "reason": "already_certified",
                        }
                    )
                    continue
                claim = _current_round_claim(
                    claims=slot_claims,
                    resource="certification",
                    firm_id=firm.firm_id,
                    order_id=order.order_id,
                    current_round=current_round,
                )
                if claim is None:
                    certification_failures.append(
                        {
                            "firm_id": firm.firm_id,
                            "order_id": order.order_id,
                            "reason": "no_current_certification_slot",
                        }
                    )
                    continue
                claim.used = True
                certified_orders.add(order.order_id)
                certification_slots_used += 1
                certification_successes.append(
                    {
                        "firm_id": firm.firm_id,
                        "order_id": order.order_id,
                        "claim_id": claim.claim_id,
                    }
                )

        successful_fulfillments: List[Dict[str, Any]] = []
        failed_fulfillments: List[Dict[str, Any]] = []
        for firm in sorted(firms, key=lambda item: item.firm_id):
            for reference in fulfillment_attempts_by_firm.get(firm.firm_id, []):
                order = _resolve_order_reference(
                    reference=reference,
                    firm_id=firm.firm_id,
                    orders_by_firm=orders_by_firm,
                )
                if order is None:
                    failed_fulfillments.append(
                        {"firm_id": firm.firm_id, "order_id": reference, "reason": "unknown_order"}
                    )
                    continue
                if current_round > order.deadline_round:
                    failed_fulfillments.append(
                        {"firm_id": firm.firm_id, "order_id": order.order_id, "reason": "past_deadline"}
                    )
                    continue
                if order.order_id not in built_orders:
                    failed_fulfillments.append(
                        {"firm_id": firm.firm_id, "order_id": order.order_id, "reason": "not_built"}
                    )
                    continue
                if order.order_id not in certified_orders:
                    failed_fulfillments.append(
                        {
                            "firm_id": firm.firm_id,
                            "order_id": order.order_id,
                            "reason": "not_certified",
                        }
                    )
                    continue
                if order.order_id in fulfilled_orders:
                    failed_fulfillments.append(
                        {
                            "firm_id": firm.firm_id,
                            "order_id": order.order_id,
                            "reason": "already_fulfilled",
                        }
                    )
                    continue
                shipping_claim = _current_round_claim(
                    claims=slot_claims,
                    resource="shipping",
                    firm_id=firm.firm_id,
                    order_id=order.order_id,
                    current_round=current_round,
                )
                if shipping_claim is None:
                    failed_fulfillments.append(
                        {
                            "firm_id": firm.firm_id,
                            "order_id": order.order_id,
                            "reason": "no_current_shipping_slot",
                        }
                    )
                    continue
                shipping_claim.used = True
                shipping_slots_used += 1
                fulfilled_orders.add(order.order_id)
                scores[firm.firm_id] += order.delivery_value
                customer_value_captured += order.delivery_value
                successful_fulfillments.append(
                    {
                        "firm_id": firm.firm_id,
                        "order_id": order.order_id,
                        "delivery_value": order.delivery_value,
                    }
                )

        public_outcomes = _public_outcomes(
            activated_certification_notice=activated_certification_notice,
            activated_shipping_notice=activated_shipping_notice,
            accepted_claims=accepted_claims_this_round,
            rejected_claims=rejected_claims_this_round,
            certification_successes=certification_successes,
            certification_failures=certification_failures,
            successful_fulfillments=successful_fulfillments,
        )

        round_log = {
            "round": current_round,
            "visible_board": {
                "notices": [_board_notice_to_dict(notice) for notice in visible_notices],
                "replies": [_board_reply_to_dict(reply) for reply in visible_replies],
            },
            "active_booking_notice_ids": {
                "certification": active_certification_notice.notice_id if active_certification_notice else None,
                "shipping": active_shipping_notice.notice_id if active_shipping_notice else None,
            },
            "certification_slot_ledger": _slot_ledger(
                claims=slot_claims,
                resource="certification",
                total_rounds=rounds,
                capacity_per_round=certification_capacity,
            ),
            "shipping_slot_ledger": _slot_ledger(
                claims=slot_claims,
                resource="shipping",
                total_rounds=rounds,
                capacity_per_round=shipping_capacity,
            ),
            "firm_plans": raw_plans_by_firm,
            "build_events": build_events,
            "build_failures": build_failures,
            "accepted_slot_claims": [_slot_claim_to_dict(claim) for claim in accepted_claims_this_round],
            "rejected_slot_claims": rejected_claims_this_round,
            "certification_successes": certification_successes,
            "certification_failures": certification_failures,
            "successful_fulfillments": successful_fulfillments,
            "failed_fulfillments": failed_fulfillments,
            "public_outcomes": public_outcomes,
            "scores": {firm_id: round(score, 4) for firm_id, score in scores.items()},
            "usage": _round_usage(usage_by_firm),
        }
        round_logs.append(round_log)

        for firm in firms:
            history_by_firm[firm.firm_id].append(
                {
                    "round": current_round,
                    "active_booking_notice_ids": round_log["active_booking_notice_ids"],
                    "public_outcomes": public_outcomes,
                    "your_plan": raw_plans_by_firm.get(firm.firm_id, {}),
                }
            )

    summary = build_summary(
        orders=orders,
        built_orders=built_orders,
        certified_orders=certified_orders,
        fulfilled_orders=fulfilled_orders,
        certification_schedule_activated=certification_schedule_activated,
        shipping_schedule_activated=shipping_schedule_activated,
        certification_notice_firm_rounds=certification_notice_firm_rounds,
        shipping_notice_firm_rounds=shipping_notice_firm_rounds,
        certification_claim_firm_rounds=certification_claim_firm_rounds,
        shipping_claim_firm_rounds=shipping_claim_firm_rounds,
        total_firm_rounds=len(firms) * rounds,
        certification_submissions=certification_submissions,
        certification_slots_used=certification_slots_used,
        shipping_slots_used=shipping_slots_used,
        total_build_cost=total_build_cost,
        customer_value_captured=customer_value_captured,
    )

    return {
        "summary": summary,
        "round_logs": round_logs,
        "orders": [
            {
                "order_id": order.order_id,
                "firm_id": order.firm_id,
                "product_name": order.product_name,
                "customer_brief": order.customer_brief,
                "delivery_value": order.delivery_value,
                "build_cost": order.build_cost,
                "deadline_round": order.deadline_round,
            }
            for order in orders
        ],
    }
