from __future__ import annotations

import json

import pytest

from experiments.escrow_market.generator import generate_firms, generate_inventory_and_orders
from experiments.escrow_market.market import _firm_private_input, run_escrow_market
from experiments.escrow_market.models import CustomerOrder, Firm, InventoryLot
from experiments.settings import ModelSettings


def _sample_firms() -> list[Firm]:
    return [
        Firm(firm_id="F1", role="buyer", system_prompt="stub"),
        Firm(firm_id="F2", role="buyer", system_prompt="stub"),
        Firm(firm_id="F3", role="seller", system_prompt="stub"),
        Firm(firm_id="F4", role="seller", system_prompt="stub"),
    ]


def _sample_lots() -> list[InventoryLot]:
    return [
        InventoryLot(
            lot_id="L1",
            seller="F3",
            quality="High",
            capability="telemetry",
            lot_name="Telemetry stack",
            description="Telemetry layer for remote monitoring and alert routing.",
            order_phrase="remote telemetry and fleet monitoring",
            reservation_price=6.0,
        ),
        InventoryLot(
            lot_id="L2",
            seller="F4",
            quality="Medium",
            capability="compliance",
            lot_name="Compliance pack",
            description="Compliance packaging and testing records.",
            order_phrase="a submission-ready compliance package",
            reservation_price=6.0,
        ),
    ]


def _sample_orders() -> list[CustomerOrder]:
    return [
        CustomerOrder(
            order_id="O1",
            buyer="F1",
            customer_brief="Deliver remote telemetry by round 3.",
            delivery_value=30.0,
            deadline_round=3,
            required_capability="telemetry",
        )
    ]


def test_generate_inventory_and_orders_is_deterministic() -> None:
    lots_a, orders_a = generate_inventory_and_orders(6, seed=77)
    lots_b, orders_b = generate_inventory_and_orders(6, seed=77)

    assert [
        (lot.lot_id, lot.seller, lot.capability, lot.lot_name, lot.quality)
        for lot in lots_a
    ] == [
        (lot.lot_id, lot.seller, lot.capability, lot.lot_name, lot.quality)
        for lot in lots_b
    ]
    assert [
        (order.order_id, order.buyer, order.required_capability, order.delivery_value)
        for order in orders_a
    ] == [
        (order.order_id, order.buyer, order.required_capability, order.delivery_value)
        for order in orders_b
    ]


def test_private_state_surface_has_board_but_no_partner_directory() -> None:
    firms = _sample_firms()
    payload = _firm_private_input(
        firm=firms[0],
        firms=firms,
        orders_by_buyer={"F1": _sample_orders()},
        acquired_lots_by_buyer={firm.firm_id: set() for firm in firms},
        lots_by_seller={"F3": [_sample_lots()[0]], "F4": [_sample_lots()[1]]},
        sold_lots=set(),
        lots_by_id={lot.lot_id: lot for lot in _sample_lots()},
        visible_notices=[],
        visible_replies=[],
        reputation={firm.firm_id: 0.0 for firm in firms},
        last_round_public_outcomes=[],
        history=[],
        score=0.0,
        rank=1,
        total_firms=len(firms),
        fulfilled_orders=set(),
        current_round=1,
        total_rounds=3,
    )

    assert "partner_directory" not in payload
    assert "module_market" not in payload
    assert "public_board" in payload
    assert "inventory_lots" in payload


def test_notice_is_not_visible_until_next_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    firms = _sample_firms()
    lots = _sample_lots()
    orders = _sample_orders()

    def fake_call_agent_json(*args, **kwargs):  # type: ignore[override]
        payload = json.loads(kwargs["user_prompt"])
        if payload["firm_id"] == "F3" and payload["current_round"] == 1:
            return (
                {
                    "public_notices": [
                        {
                            "headline": "Telemetry stack available",
                            "body": "We can provide remote telemetry and fleet monitoring.",
                            "cash_terms": "Minimum 6.0 cash.",
                        }
                    ],
                    "reply_notices": [],
                    "fulfill_orders": [],
                    "commentary": "",
                },
                '{"stub": true}',
            )
        return (
            {
                "public_notices": [],
                "reply_notices": [],
                "fulfill_orders": [],
                "commentary": "",
            },
            '{"stub": true}',
        )

    monkeypatch.setattr("experiments.escrow_market.market.call_agent_json", fake_call_agent_json)

    result = run_escrow_market(
        model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        firms=firms,
        lots=lots,
        orders=orders,
        rounds=2,
        seed=10,
    )

    assert result["round_logs"][0]["visible_board"]["notices"] == []
    assert result["round_logs"][1]["visible_board"]["notices"][0]["notice_id"] == "N1"


def test_deal_needs_visible_offer_notice_and_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    firms = _sample_firms()
    lots = _sample_lots()
    orders = _sample_orders()

    def fake_call_agent_json(*args, **kwargs):  # type: ignore[override]
        payload = json.loads(kwargs["user_prompt"])
        if payload["firm_id"] == "F1":
            return (
                {
                    "public_notices": [],
                    "reply_notices": [
                        {
                            "notice_id": "N1",
                            "body": "Want telemetry for order O1.",
                            "cash_offer": 8.0,
                        }
                    ],
                    "fulfill_orders": ["O1"],
                    "commentary": "",
                },
                '{"stub": true}',
            )
        return (
            {
                "public_notices": [],
                "reply_notices": [],
                "fulfill_orders": [],
                "commentary": "",
            },
            '{"stub": true}',
        )

    monkeypatch.setattr("experiments.escrow_market.market.call_agent_json", fake_call_agent_json)

    result = run_escrow_market(
        model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        firms=firms,
        lots=lots,
        orders=orders,
        rounds=1,
        seed=12,
    )

    assert result["summary"]["deal_volume"] == 0
    assert result["summary"]["orders_fulfilled"] == 0


def test_institution_can_activate_from_notice_and_support_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    firms = _sample_firms()
    lots = _sample_lots()
    orders = _sample_orders()

    def fake_call_agent_json(*args, **kwargs):  # type: ignore[override]
        payload = json.loads(kwargs["user_prompt"])
        firm_id = payload["firm_id"]
        current_round = payload["current_round"]
        if firm_id == "F1" and current_round == 1:
            return (
                {
                    "public_notices": [
                        {
                            "headline": "Escrow process proposal",
                            "body": "Use shared escrow to hold payment until delivery works.",
                            "cash_terms": "",
                        }
                    ],
                    "reply_notices": [],
                    "fulfill_orders": [],
                    "commentary": "",
                },
                '{"stub": true}',
            )
        if firm_id == "F2" and current_round == 2:
            return (
                {
                    "public_notices": [],
                    "reply_notices": [
                        {
                            "notice_id": "N1",
                            "body": "Agree to use this escrow process.",
                            "cash_offer": 0.0,
                        }
                    ],
                    "fulfill_orders": [],
                    "commentary": "",
                },
                '{"stub": true}',
            )
        return (
            {
                "public_notices": [],
                "reply_notices": [],
                "fulfill_orders": [],
                "commentary": "",
            },
            '{"stub": true}',
        )

    monkeypatch.setattr("experiments.escrow_market.market.call_agent_json", fake_call_agent_json)

    result = run_escrow_market(
        model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        firms=firms,
        lots=lots,
        orders=orders,
        rounds=2,
        seed=14,
    )

    assert result["summary"]["institutions_activated"] == 1
    assert result["round_logs"][1]["institutions_activated"] == [
        {
            "notice_id": "N1",
            "founder": "F1",
            "supporter": "F2",
            "activated_round": 2,
        }
    ]


def test_supported_escrow_can_protect_trade_and_fulfillment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    firms = _sample_firms()
    lots = _sample_lots()
    orders = _sample_orders()

    def fake_call_agent_json(*args, **kwargs):  # type: ignore[override]
        payload = json.loads(kwargs["user_prompt"])
        firm_id = payload["firm_id"]
        current_round = payload["current_round"]
        if firm_id == "F1" and current_round == 1:
            return (
                {
                    "public_notices": [
                        {
                            "headline": "Escrow process proposal",
                            "body": "Use shared escrow to hold payment until delivery works.",
                            "cash_terms": "",
                        }
                    ],
                    "reply_notices": [],
                    "fulfill_orders": [],
                    "commentary": "",
                },
                '{"stub": true}',
            )
        if firm_id == "F3" and current_round == 1:
            return (
                {
                    "public_notices": [
                        {
                            "headline": "Telemetry stack available",
                            "body": "We can provide remote telemetry and fleet monitoring.",
                            "cash_terms": "Minimum 6.0 cash.",
                        }
                    ],
                    "reply_notices": [],
                    "fulfill_orders": [],
                    "commentary": "",
                },
                '{"stub": true}',
            )
        if firm_id == "F2" and current_round == 2:
            return (
                {
                    "public_notices": [],
                    "reply_notices": [
                        {
                            "notice_id": "N1",
                            "body": "Agree to use this escrow process.",
                            "cash_offer": 0.0,
                        }
                    ],
                    "fulfill_orders": [],
                    "commentary": "",
                },
                '{"stub": true}',
            )
        if firm_id == "F1" and current_round == 2:
            return (
                {
                    "public_notices": [],
                    "reply_notices": [
                        {
                            "notice_id": "N2",
                            "body": "Need telemetry for order O1.",
                            "cash_offer": 8.0,
                        }
                    ],
                    "fulfill_orders": ["O1"],
                    "commentary": "",
                },
                '{"stub": true}',
            )
        return (
            {
                "public_notices": [],
                "reply_notices": [],
                "fulfill_orders": [],
                "commentary": "",
            },
            '{"stub": true}',
        )

    monkeypatch.setattr("experiments.escrow_market.market.call_agent_json", fake_call_agent_json)

    result = run_escrow_market(
        model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        firms=firms,
        lots=lots,
        orders=orders,
        rounds=2,
        seed=18,
    )

    assert result["summary"]["institutions_activated"] == 1
    assert result["summary"]["safe_deal_volume"] == 1
    assert result["summary"]["orders_fulfilled"] == 1


def test_dry_run_keeps_board_empty_and_zero_deals() -> None:
    lots, orders = generate_inventory_and_orders(6, seed=9)
    firms = generate_firms(6, arm="baseline")
    result = run_escrow_market(
        model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        firms=firms,
        lots=lots,
        orders=orders,
        dry_run=True,
        rounds=1,
        seed=9,
    )

    plans = result["round_logs"][0]["plans"]
    assert all(plan["public_notices"] == [] for plan in plans.values())
    assert all(plan["reply_notices"] == [] for plan in plans.values())
    assert result["summary"]["deal_volume"] == 0
