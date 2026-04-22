from __future__ import annotations

import json

import pytest

from experiments.ip_brokered_market.generator import generate_firms, generate_modules_and_orders
from experiments.ip_brokered_market.market import _firm_private_input, run_brokered_market
from experiments.ip_brokered_market.models import CustomerOrder, Firm, Module
from experiments.settings import ModelSettings


def _sample_firms() -> list[Firm]:
    return [Firm(firm_id=f"F{i+1}", system_prompt="stub") for i in range(4)]


def _sample_modules(firms: list[Firm]) -> list[Module]:
    return [
        Module(
            tech_id="T1",
            owner="F1",
            quality="High",
            capability="control",
            module_name="Control core",
            description="Control firmware for connected devices.",
            order_phrase="reliable device control firmware",
            reference_license_price=5.0,
            integration_costs={firm.firm_id: 1.0 for firm in firms},
            substitute_costs={firm.firm_id: 9.0 for firm in firms},
        ),
        Module(
            tech_id="T2",
            owner="F2",
            quality="High",
            capability="telemetry",
            module_name="Telemetry stack",
            description="Telemetry layer for remote monitoring and alert routing.",
            order_phrase="remote telemetry and fleet monitoring",
            reference_license_price=5.0,
            integration_costs={firm.firm_id: 1.0 for firm in firms},
            substitute_costs={firm.firm_id: 8.0 for firm in firms},
        ),
        Module(
            tech_id="T3",
            owner="F3",
            quality="Medium",
            capability="compliance",
            module_name="Compliance pack",
            description="Compliance packaging and testing records.",
            order_phrase="a submission-ready compliance package",
            reference_license_price=5.0,
            integration_costs={firm.firm_id: 1.0 for firm in firms},
            substitute_costs={firm.firm_id: 8.0 for firm in firms},
        ),
        Module(
            tech_id="T4",
            owner="F4",
            quality="Medium",
            capability="security",
            module_name="Security bundle",
            description="Authentication, audit, and secure update components.",
            order_phrase="enterprise security hardening",
            reference_license_price=5.0,
            integration_costs={firm.firm_id: 1.0 for firm in firms},
            substitute_costs={firm.firm_id: 8.0 for firm in firms},
        ),
    ]


def _sample_orders() -> list[CustomerOrder]:
    return [
        CustomerOrder(
            order_id="O1",
            target_firm="F1",
            customer_brief="Deliver one package that covers device control and remote telemetry.",
            delivery_value=32.0,
            deadline_round=3,
            required_capabilities=["control", "telemetry"],
        )
    ]


def test_generate_modules_and_orders_is_deterministic() -> None:
    modules_a, orders_a = generate_modules_and_orders(6, 12, 6, seed=77)
    modules_b, orders_b = generate_modules_and_orders(6, 12, 6, seed=77)

    assert [
        (module.tech_id, module.owner, module.capability, module.module_name)
        for module in modules_a
    ] == [
        (module.tech_id, module.owner, module.capability, module.module_name)
        for module in modules_b
    ]
    assert [
        (order.order_id, order.target_firm, tuple(order.required_capabilities), order.delivery_value)
        for order in orders_a
    ] == [
        (order.order_id, order.target_firm, tuple(order.required_capabilities), order.delivery_value)
        for order in orders_b
    ]


def test_private_state_surface_hides_direct_counterparty_lookup() -> None:
    firms = _sample_firms()
    modules = _sample_modules(firms)
    payload = _firm_private_input(
        firm=firms[0],
        firms=firms,
        modules=modules,
        orders=_sample_orders(),
        holdings_by_firm={"F1": {"T1"}, "F2": {"T2"}, "F3": {"T3"}, "F4": {"T4"}},
        completed_internal_capabilities_by_firm={firm.firm_id: set() for firm in firms},
        internal_projects_by_firm={firm.firm_id: [] for firm in firms},
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
    assert "partner_requests" not in json.dumps(payload)
    assert "public_board" in payload
    assert "public_reputation" in payload


def test_notice_post_is_not_visible_until_next_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    firms = _sample_firms()
    modules = _sample_modules(firms)
    orders = _sample_orders()

    def fake_call_agent_json(*args, **kwargs):  # type: ignore[override]
        payload = json.loads(kwargs["user_prompt"])
        if payload["firm_id"] == "F2" and payload["current_round"] == 1:
            return (
                {
                    "internal_projects": [],
                    "public_notices": [
                        {
                            "headline": "Telemetry support available",
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
                "internal_projects": [],
                "public_notices": [],
                "reply_notices": [],
                "fulfill_orders": [],
                "commentary": "",
            },
            '{"stub": true}',
        )

    monkeypatch.setattr(
        "experiments.ip_brokered_market.market.call_agent_json",
        fake_call_agent_json,
    )

    result = run_brokered_market(
        model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        firms=firms,
        modules=modules,
        orders=orders,
        rounds=2,
        seed=11,
    )

    assert result["round_logs"][0]["visible_board"]["notices"] == []
    assert result["round_logs"][1]["visible_board"]["notices"] == [
        {
            "notice_id": "N1",
            "author": "F2",
            "headline": "Telemetry support available",
            "body": "We can provide remote telemetry and fleet monitoring.",
            "cash_terms": "Minimum 6.0 cash.",
            "round_posted": 1,
        }
    ]


def test_deal_does_not_clear_without_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    firms = _sample_firms()
    modules = _sample_modules(firms)
    orders = _sample_orders()

    def fake_call_agent_json(*args, **kwargs):  # type: ignore[override]
        payload = json.loads(kwargs["user_prompt"])
        if payload["firm_id"] == "F2":
            return (
                {
                    "internal_projects": [],
                    "public_notices": [
                        {
                            "headline": "Telemetry support available",
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
                "internal_projects": [],
                "public_notices": [],
                "reply_notices": [],
                "fulfill_orders": [],
                "commentary": "",
            },
            '{"stub": true}',
        )

    monkeypatch.setattr(
        "experiments.ip_brokered_market.market.call_agent_json",
        fake_call_agent_json,
    )

    result = run_brokered_market(
        model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        firms=firms,
        modules=modules,
        orders=orders,
        rounds=1,
        seed=14,
    )

    assert result["summary"]["brokered_deal_volume"] == 0
    assert result["summary"]["orders_fulfilled"] == 0


def test_visible_offer_notice_and_reply_can_clear_brokered_deal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    firms = _sample_firms()
    modules = _sample_modules(firms)
    orders = _sample_orders()

    def fake_call_agent_json(*args, **kwargs):  # type: ignore[override]
        payload = json.loads(kwargs["user_prompt"])
        firm_id = payload["firm_id"]
        current_round = payload["current_round"]
        if firm_id == "F2" and current_round == 1:
            return (
                {
                    "internal_projects": [],
                    "public_notices": [
                        {
                            "headline": "Telemetry stack available",
                            "body": "We can provide remote telemetry and fleet monitoring right away.",
                            "cash_terms": "Minimum 6.0 cash.",
                        }
                    ],
                    "reply_notices": [],
                    "fulfill_orders": [],
                    "commentary": "",
                },
                '{"stub": true}',
            )
        if firm_id == "F1" and current_round == 2:
            return (
                {
                    "internal_projects": [],
                    "public_notices": [],
                    "reply_notices": [
                        {
                            "notice_id": "N1",
                            "body": "Need telemetry support for order O1.",
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
                "internal_projects": [],
                "public_notices": [],
                "reply_notices": [],
                "fulfill_orders": [],
                "commentary": "",
            },
            '{"stub": true}',
        )

    monkeypatch.setattr(
        "experiments.ip_brokered_market.market.call_agent_json",
        fake_call_agent_json,
    )

    result = run_brokered_market(
        model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        firms=firms,
        modules=modules,
        orders=orders,
        rounds=2,
        seed=21,
    )

    assert result["summary"]["brokered_deal_volume"] == 1
    assert result["summary"]["orders_fulfilled"] == 1
    assert "T2" in result["brokered_modules_by_firm"]["F1"]


def test_ambiguous_notice_text_does_not_clear_deal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    firms = _sample_firms()
    modules = _sample_modules(firms)
    orders = _sample_orders()

    def fake_call_agent_json(*args, **kwargs):  # type: ignore[override]
        payload = json.loads(kwargs["user_prompt"])
        firm_id = payload["firm_id"]
        current_round = payload["current_round"]
        if firm_id == "F2" and current_round == 1:
            return (
                {
                    "internal_projects": [],
                    "public_notices": [
                        {
                            "headline": "Available support",
                            "body": "We can provide one component for the right project.",
                            "cash_terms": "Minimum 6.0 cash.",
                        }
                    ],
                    "reply_notices": [],
                    "fulfill_orders": [],
                    "commentary": "",
                },
                '{"stub": true}',
            )
        if firm_id == "F1" and current_round == 2:
            return (
                {
                    "internal_projects": [],
                    "public_notices": [],
                    "reply_notices": [
                        {
                            "notice_id": "N1",
                            "body": "Need outside help for order O1.",
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
                "internal_projects": [],
                "public_notices": [],
                "reply_notices": [],
                "fulfill_orders": [],
                "commentary": "",
            },
            '{"stub": true}',
        )

    monkeypatch.setattr(
        "experiments.ip_brokered_market.market.call_agent_json",
        fake_call_agent_json,
    )

    result = run_brokered_market(
        model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        firms=firms,
        modules=modules,
        orders=orders,
        rounds=2,
        seed=25,
    )

    assert result["summary"]["brokered_deal_volume"] == 0
    assert result["round_logs"][1]["unresolved_replies"] == [
        {
            "reply_id": "R1",
            "notice_id": "N1",
            "buyer": "F1",
            "seller": "F2",
            "reason": "ambiguous_notice",
        }
    ]


def test_internal_project_started_in_round_one_completes_in_round_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    firms = _sample_firms()
    modules = _sample_modules(firms)
    orders = _sample_orders()
    call_counts = {"F1": 0}

    def fake_call_agent_json(*args, **kwargs):  # type: ignore[override]
        payload = json.loads(kwargs["user_prompt"])
        firm_id = payload["firm_id"]
        if firm_id != "F1":
            return (
                {
                    "internal_projects": [],
                    "public_notices": [],
                    "reply_notices": [],
                    "fulfill_orders": [],
                    "commentary": "",
                },
                '{"stub": true}',
            )
        call_counts["F1"] += 1
        if call_counts["F1"] == 1:
            return (
                {
                    "internal_projects": [{"objective": "Build remote telemetry and fleet monitoring in-house."}],
                    "public_notices": [],
                    "reply_notices": [],
                    "fulfill_orders": ["O1"],
                    "commentary": "",
                },
                '{"stub": true}',
            )
        return (
            {
                "internal_projects": [],
                "public_notices": [],
                "reply_notices": [],
                "fulfill_orders": ["O1"],
                "commentary": "",
            },
            '{"stub": true}',
        )

    monkeypatch.setattr(
        "experiments.ip_brokered_market.market.call_agent_json",
        fake_call_agent_json,
    )

    result = run_brokered_market(
        model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        firms=firms,
        modules=modules,
        orders=orders,
        rounds=3,
        seed=12,
    )

    assert result["round_logs"][0]["successful_fulfillments"] == []
    assert result["round_logs"][1]["successful_fulfillments"] == []
    assert result["round_logs"][2]["successful_fulfillments"] == [
        {"firm_id": "F1", "order_id": "O1", "delivery_value": 32.0}
    ]
    assert result["completed_internal_capabilities_by_firm"]["F1"] == ["telemetry"]
    assert "T2" not in result["brokered_modules_by_firm"]["F1"]


def test_dry_run_keeps_board_empty_and_zero_deals() -> None:
    modules, orders = generate_modules_and_orders(4, 4, 4, seed=9)
    firms = generate_firms(4, arm="baseline")
    result = run_brokered_market(
        model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        firms=firms,
        modules=modules,
        orders=orders,
        dry_run=True,
        rounds=1,
        seed=9,
    )

    plans = result["round_logs"][0]["plans"]
    assert all(plan["internal_projects"] == [] for plan in plans.values())
    assert all(plan["public_notices"] == [] for plan in plans.values())
    assert all(plan["reply_notices"] == [] for plan in plans.values())
    assert result["summary"]["brokered_deal_volume"] == 0
