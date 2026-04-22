from __future__ import annotations

import json

import pytest

from experiments.certification_market.generator import generate_firms, generate_orders
from experiments.certification_market.market import _firm_private_input, run_certification_market
from experiments.certification_market.models import CustomerOrder, Firm
from experiments.settings import ModelSettings


def _sample_firms() -> list[Firm]:
    return [
        Firm(firm_id="F1", system_prompt="stub"),
        Firm(firm_id="F2", system_prompt="stub"),
    ]


def _sample_orders() -> list[CustomerOrder]:
    return [
        CustomerOrder(
            order_id="O1",
            firm_id="F1",
            product_name="fleet telemetry relay",
            customer_brief="Deliver one certified and shipped relay by round 4.",
            delivery_value=28.0,
            build_cost=6.0,
            deadline_round=4,
        ),
        CustomerOrder(
            order_id="O2",
            firm_id="F2",
            product_name="factory safety sensor",
            customer_brief="Deliver one certified and shipped sensor by round 4.",
            delivery_value=27.0,
            build_cost=6.5,
            deadline_round=4,
        ),
    ]


def test_generate_orders_is_deterministic() -> None:
    orders_a = generate_orders(4, seed=77)
    orders_b = generate_orders(4, seed=77)

    assert [
        (order.order_id, order.firm_id, order.product_name, order.delivery_value, order.build_cost)
        for order in orders_a
    ] == [
        (order.order_id, order.firm_id, order.product_name, order.delivery_value, order.build_cost)
        for order in orders_b
    ]


def test_private_state_surface_shows_both_shared_resources() -> None:
    firms = _sample_firms()
    orders = _sample_orders()
    payload = _firm_private_input(
        firm=firms[0],
        orders_by_firm={"F1": [orders[0]], "F2": [orders[1]]},
        built_orders=set(),
        certified_orders=set(),
        fulfilled_orders=set(),
        visible_notices=[],
        visible_replies=[],
        active_certification_notice=None,
        active_shipping_notice=None,
        slot_claims=[],
        history=[],
        score=0.0,
        rank=1,
        total_firms=len(firms),
        current_round=1,
        total_rounds=4,
        certification_capacity=2,
        shipping_capacity=2,
    )

    assert "partner_directory" not in payload
    assert "certification_lab" in payload
    assert "shipping_dock" in payload
    assert payload["customer_orders"][0]["product_name"] == "fleet telemetry relay"


def test_schedule_notice_is_not_visible_until_next_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    firms = _sample_firms()
    orders = _sample_orders()

    def fake_call_agent_json(*args, **kwargs):  # type: ignore[override]
        payload = json.loads(kwargs["user_prompt"])
        if payload["firm_id"] == "F1" and payload["current_round"] == 1:
            return (
                {
                    "build_products": [],
                    "public_notices": [
                        {
                            "headline": "Certification booking schedule",
                            "body": "Post slot requests here so we can schedule the lab.",
                        }
                    ],
                    "reply_notices": [],
                    "submit_for_certification": [],
                    "fulfill_orders": [],
                    "commentary": "",
                },
                '{"stub": true}',
            )
        return (
            {
                "build_products": [],
                "public_notices": [],
                "reply_notices": [],
                "submit_for_certification": [],
                "fulfill_orders": [],
                "commentary": "",
            },
            '{"stub": true}',
        )

    monkeypatch.setattr(
        "experiments.certification_market.market.call_agent_json",
        fake_call_agent_json,
    )

    result = run_certification_market(
        model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        firms=firms,
        orders=orders,
        rounds=2,
        certification_capacity=2,
        shipping_capacity=2,
    )

    assert result["round_logs"][0]["visible_board"]["notices"] == []
    assert result["round_logs"][1]["visible_board"]["notices"][0]["notice_id"] == "N1"


def test_certification_needs_current_round_slot_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    firms = _sample_firms()
    orders = _sample_orders()

    def fake_call_agent_json(*args, **kwargs):  # type: ignore[override]
        payload = json.loads(kwargs["user_prompt"])
        if payload["firm_id"] == "F1":
            return (
                {
                    "build_products": ["O1"],
                    "public_notices": [],
                    "reply_notices": [],
                    "submit_for_certification": ["O1"],
                    "fulfill_orders": ["O1"],
                    "commentary": "",
                },
                '{"stub": true}',
            )
        return (
            {
                "build_products": [],
                "public_notices": [],
                "reply_notices": [],
                "submit_for_certification": [],
                "fulfill_orders": [],
                "commentary": "",
            },
            '{"stub": true}',
        )

    monkeypatch.setattr(
        "experiments.certification_market.market.call_agent_json",
        fake_call_agent_json,
    )

    result = run_certification_market(
        model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        firms=firms,
        orders=orders,
        rounds=1,
        certification_capacity=2,
        shipping_capacity=2,
    )

    assert result["summary"]["products_built"] == 1
    assert result["summary"]["certification_successes"] == 0
    assert result["summary"]["orders_fulfilled"] == 0
    assert (
        result["round_logs"][0]["certification_failures"][0]["reason"]
        == "no_current_certification_slot"
    )


def test_chain_needs_both_certification_and_shipping_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    firms = _sample_firms()
    orders = _sample_orders()

    def fake_call_agent_json(*args, **kwargs):  # type: ignore[override]
        payload = json.loads(kwargs["user_prompt"])
        firm_id = payload["firm_id"]
        current_round = payload["current_round"]
        if firm_id == "F1" and current_round == 1:
            return (
                {
                    "build_products": ["fleet telemetry relay"],
                    "public_notices": [
                        {
                            "headline": "Certification schedule",
                            "body": "Use this board to reserve lab slots by round.",
                        },
                        {
                            "headline": "Shipping dock schedule",
                            "body": "Use this board to reserve shipping dock slots by round.",
                        },
                    ],
                    "reply_notices": [],
                    "submit_for_certification": [],
                    "fulfill_orders": [],
                    "commentary": "",
                },
                '{"stub": true}',
            )
        if firm_id == "F1" and current_round == 2:
            return (
                {
                    "build_products": [],
                    "public_notices": [],
                    "reply_notices": [
                        {
                            "notice_id": "N1",
                            "body": "Reserve fleet telemetry relay for round 2.",
                        }
                    ],
                    "submit_for_certification": ["fleet telemetry relay"],
                    "fulfill_orders": [],
                    "commentary": "",
                },
                '{"stub": true}',
            )
        if firm_id == "F1" and current_round == 3:
            return (
                {
                    "build_products": [],
                    "public_notices": [],
                    "reply_notices": [
                        {
                            "notice_id": "N2",
                            "body": "Reserve fleet telemetry relay for round 3.",
                        }
                    ],
                    "submit_for_certification": [],
                    "fulfill_orders": ["fleet telemetry relay"],
                    "commentary": "",
                },
                '{"stub": true}',
            )
        return (
            {
                "build_products": [],
                "public_notices": [],
                "reply_notices": [],
                "submit_for_certification": [],
                "fulfill_orders": [],
                "commentary": "",
            },
            '{"stub": true}',
        )

    monkeypatch.setattr(
        "experiments.certification_market.market.call_agent_json",
        fake_call_agent_json,
    )

    result = run_certification_market(
        model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        firms=firms,
        orders=orders,
        rounds=3,
        certification_capacity=2,
        shipping_capacity=2,
    )

    assert result["summary"]["certification_schedule_activation_rate"] == 1.0
    assert result["summary"]["shipping_schedule_activation_rate"] == 1.0
    assert result["summary"]["certification_successes"] == 1
    assert result["summary"]["orders_fulfilled"] == 1


def test_product_name_alias_is_accepted_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    firms = _sample_firms()
    orders = _sample_orders()

    def fake_call_agent_json(*args, **kwargs):  # type: ignore[override]
        payload = json.loads(kwargs["user_prompt"])
        firm_id = payload["firm_id"]
        current_round = payload["current_round"]
        if firm_id == "F1" and current_round == 1:
            return (
                {
                    "build_products": ["fleet telemetry relay"],
                    "public_notices": [
                        {
                            "headline": "Certification schedule",
                            "body": "Reserve lab slots here.",
                        },
                        {
                            "headline": "Shipping schedule",
                            "body": "Reserve dock slots here.",
                        },
                    ],
                    "reply_notices": [],
                    "submit_for_certification": [],
                    "fulfill_orders": [],
                    "commentary": "",
                },
                '{"stub": true}',
            )
        if firm_id == "F1" and current_round == 2:
            return (
                {
                    "build_products": [],
                    "public_notices": [],
                    "reply_notices": [
                        {"notice_id": "N1", "body": "Reserve fleet telemetry relay for round 2."},
                        {"notice_id": "N2", "body": "Reserve fleet telemetry relay for round 3."},
                    ],
                    "submit_for_certification": ["fleet telemetry relay"],
                    "fulfill_orders": [],
                    "commentary": "",
                },
                '{"stub": true}',
            )
        if firm_id == "F1" and current_round == 3:
            return (
                {
                    "build_products": [],
                    "public_notices": [],
                    "reply_notices": [],
                    "submit_for_certification": [],
                    "fulfill_orders": ["fleet telemetry relay"],
                    "commentary": "",
                },
                '{"stub": true}',
            )
        return (
            {
                "build_products": [],
                "public_notices": [],
                "reply_notices": [],
                "submit_for_certification": [],
                "fulfill_orders": [],
                "commentary": "",
            },
            '{"stub": true}',
        )

    monkeypatch.setattr(
        "experiments.certification_market.market.call_agent_json",
        fake_call_agent_json,
    )

    result = run_certification_market(
        model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        firms=firms,
        orders=orders,
        rounds=3,
        certification_capacity=2,
        shipping_capacity=2,
    )

    assert result["summary"]["products_built"] == 1
    assert result["summary"]["orders_fulfilled"] == 1


def test_dry_run_preserves_empty_actions() -> None:
    firms = generate_firms(4, arm="baseline")
    orders = generate_orders(4, seed=14)

    result = run_certification_market(
        model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        firms=firms,
        orders=orders,
        dry_run=True,
        rounds=2,
        certification_capacity=2,
        shipping_capacity=2,
    )

    assert result["summary"]["products_built"] == 0
    assert result["summary"]["orders_fulfilled"] == 0
    assert result["summary"]["certification_schedule_activation_rate"] == 0.0
    assert result["summary"]["shipping_schedule_activation_rate"] == 0.0
