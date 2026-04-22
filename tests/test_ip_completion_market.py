from __future__ import annotations

import json

import pytest

from experiments.ip_completion_market.generator import generate_firms, generate_modules_and_orders
from experiments.ip_completion_market.market import (
    build_summary,
    evaluate_fulfillment_attempts,
    order_state_snapshot,
    run_completion_market,
    started_not_delivered_orders,
)
from experiments.ip_completion_market.models import CustomerOrder, Firm, Module
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
            substitute_costs={firm.firm_id: 8.0 for firm in firms},
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
            substitute_costs={firm.firm_id: 7.0 for firm in firms},
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
            substitute_costs={firm.firm_id: 7.0 for firm in firms},
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
            substitute_costs={firm.firm_id: 7.0 for firm in firms},
        ),
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


def test_order_fulfillment_requires_all_capabilities() -> None:
    order = CustomerOrder(
        order_id="O1",
        target_firm="F1",
        customer_brief="Deliver a package with telemetry, control, and compliance.",
        delivery_value=32.0,
        deadline_round=2,
        required_capabilities=["telemetry", "control", "compliance"],
    )
    successful, failed, value = evaluate_fulfillment_attempts(
        orders_by_id={"O1": order},
        available_capabilities_by_firm={"F1": {"telemetry", "control"}},
        fulfillment_attempts_by_firm={"F1": ["O1"]},
        fulfilled_orders=set(),
        current_round=1,
    )

    assert successful == []
    assert failed == [
        {
            "firm_id": "F1",
            "order_id": "O1",
            "reason": "missing_capabilities",
            "missing_capabilities": ["compliance"],
        }
    ]
    assert value == 0.0


def test_started_not_delivered_metric_tracks_partial_progress() -> None:
    order = CustomerOrder(
        order_id="O1",
        target_firm="F1",
        customer_brief="Deliver telemetry and compliance in one package.",
        delivery_value=30.0,
        deadline_round=2,
        required_capabilities=["telemetry", "compliance"],
    )
    order_state = order_state_snapshot(
        orders=[order],
        available_capabilities_by_firm={"F1": {"telemetry"}},
        started_orders={"O1"},
        fulfilled_orders=set(),
        substitute_projects_by_firm={"F1": []},
    )
    summary = build_summary(
        orders=[order],
        order_state=order_state,
        total_deal_volume=1,
        total_partner_requests=1,
        total_substitute_builds_started=1,
        total_substitute_builds_completed=0,
        total_welfare=9.5,
        customer_value_captured=0.0,
        partner_request_firm_rounds=1,
        total_firm_rounds=2,
    )

    assert started_not_delivered_orders(order_state) == ["O1"]
    assert summary["started_not_delivered_count"] == 1
    assert summary["customer_value_captured"] == 0.0
    assert summary["partner_request_rate"] == 0.5


def test_dry_run_keeps_empty_partner_requests() -> None:
    modules, orders = generate_modules_and_orders(4, 4, 4, seed=9)
    firms = generate_firms(4, arm="baseline")
    result = run_completion_market(
        model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        firms=firms,
        modules=modules,
        orders=orders,
        dry_run=True,
        rounds=1,
        seed=9,
        scenario="partner_directory",
    )

    plans = result["round_logs"][0]["plans"]
    assert all(plan["internal_projects"] == [] for plan in plans.values())
    assert all(plan["partner_requests"] == [] for plan in plans.values())
    assert result["summary"]["deal_volume"] == 0


def test_partner_request_can_execute_deal_without_named_buy_sell_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    firms = _sample_firms()
    modules = _sample_modules(firms)
    orders = [
        CustomerOrder(
            order_id="O1",
            target_firm="F1",
            customer_brief="Deliver one package that covers device control and remote telemetry.",
            delivery_value=32.0,
            deadline_round=1,
            required_capabilities=["control", "telemetry"],
        )
    ]

    def fake_call_agent_json(*args, **kwargs):  # type: ignore[override]
        payload = json.loads(kwargs["user_prompt"])
        if payload["firm_id"] != "F1":
            return (
                {
                    "internal_projects": [],
                    "partner_requests": [],
                    "fulfill_orders": [],
                    "commentary": "",
                },
                '{"stub": true}',
            )
        return (
            {
                "internal_projects": [],
                "partner_requests": [
                    {
                        "to_firm": "F2",
                        "request": "Need remote telemetry and fleet monitoring for order O1.",
                        "cash_offer": 8.0,
                        "note": "Can move immediately.",
                    }
                ],
                "fulfill_orders": ["O1"],
                "commentary": "Request outside help for telemetry.",
            },
            '{"stub": true}',
        )

    monkeypatch.setattr(
        "experiments.ip_completion_market.market.call_agent_json",
        fake_call_agent_json,
    )

    result = run_completion_market(
        model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        firms=firms,
        modules=modules,
        orders=orders,
        rounds=1,
        seed=21,
        scenario="partner_directory",
    )

    assert result["summary"]["deal_volume"] == 1
    assert result["summary"]["orders_fulfilled"] == 1
    assert "T2" in result["licensed_modules_by_firm"]["F1"]


def test_substitute_started_in_round_one_completes_in_round_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    firms = _sample_firms()
    modules = _sample_modules(firms)
    orders = [
        CustomerOrder(
            order_id="O1",
            target_firm="F1",
            customer_brief="Deliver control and telemetry together.",
            delivery_value=32.0,
            deadline_round=2,
            required_capabilities=["control", "telemetry"],
        )
    ]

    call_counts = {"F1": 0}

    def fake_call_agent_json(*args, **kwargs):  # type: ignore[override]
        payload = json.loads(kwargs["user_prompt"])
        firm_id = payload["firm_id"]
        if firm_id != "F1":
            return (
                {
                    "internal_projects": [],
                    "partner_requests": [],
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
                    "partner_requests": [],
                    "fulfill_orders": ["O1"],
                    "commentary": "Start substitute.",
                },
                '{"stub": true}',
            )
        return (
            {
                "internal_projects": [],
                "partner_requests": [],
                "fulfill_orders": ["O1"],
                "commentary": "Deliver now.",
            },
            '{"stub": true}',
        )

    monkeypatch.setattr(
        "experiments.ip_completion_market.market.call_agent_json",
        fake_call_agent_json,
    )

    result = run_completion_market(
        model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        firms=firms,
        modules=modules,
        orders=orders,
        rounds=2,
        seed=12,
        scenario="partner_directory",
    )

    assert result["round_logs"][0]["successful_fulfillments"] == []
    assert result["round_logs"][1]["successful_fulfillments"] == [
        {"firm_id": "F1", "order_id": "O1", "delivery_value": 32.0}
    ]
    assert result["completed_substitutes_by_firm"]["F1"] == ["telemetry"]
    assert "T2" not in result["licensed_modules_by_firm"]["F1"]


def test_firm_cannot_build_rival_module_by_naming_module_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    firms = _sample_firms()
    modules = _sample_modules(firms)
    orders = [
        CustomerOrder(
            order_id="O1",
            target_firm="F1",
            customer_brief="Deliver control and telemetry together.",
            delivery_value=32.0,
            deadline_round=2,
            required_capabilities=["control", "telemetry"],
        )
    ]

    def fake_call_agent_json(*args, **kwargs):  # type: ignore[override]
        payload = json.loads(kwargs["user_prompt"])
        if payload["firm_id"] != "F1":
            return (
                {
                    "internal_projects": [],
                    "partner_requests": [],
                    "fulfill_orders": [],
                    "commentary": "",
                },
                '{"stub": true}',
            )
        return (
            {
                "internal_projects": [{"objective": "T2"}],
                "partner_requests": [],
                "fulfill_orders": ["O1"],
                "commentary": "Try to copy rival module by id.",
            },
            '{"stub": true}',
        )

    monkeypatch.setattr(
        "experiments.ip_completion_market.market.call_agent_json",
        fake_call_agent_json,
    )

    result = run_completion_market(
        model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        firms=firms,
        modules=modules,
        orders=orders,
        rounds=2,
        seed=33,
        scenario="partner_directory",
    )

    assert result["completed_substitutes_by_firm"]["F1"] == []
    assert result["summary"]["orders_fulfilled"] == 0
