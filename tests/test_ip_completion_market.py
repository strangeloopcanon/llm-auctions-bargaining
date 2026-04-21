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
        total_substitute_builds_started=1,
        total_substitute_builds_completed=0,
        total_welfare=9.5,
        customer_value_captured=0.0,
        trade_active_firm_rounds=1,
        total_firm_rounds=2,
    )

    assert started_not_delivered_orders(order_state) == ["O1"]
    assert summary["started_not_delivered_count"] == 1
    assert summary["customer_value_captured"] == 0.0


def test_dry_run_keeps_empty_trade_proposals(monkeypatch: pytest.MonkeyPatch) -> None:
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
    )

    plans = result["round_logs"][0]["plans"]
    assert all(plan["license_buys"] == [] for plan in plans.values())
    assert all(plan["license_sells"] == [] for plan in plans.values())
    assert result["summary"]["deal_volume"] == 0


def test_substitute_started_in_round_one_completes_in_round_two(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    firms = [Firm(firm_id=f"F{i+1}", system_prompt="stub") for i in range(4)]
    modules = [
        Module(
            tech_id="T1",
            owner="F1",
            quality="High",
            capability="control",
            module_name="Control core",
            description="Control core",
            order_phrase="control",
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
            description="Telemetry stack",
            order_phrase="telemetry",
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
            description="Compliance pack",
            order_phrase="compliance",
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
            description="Security bundle",
            order_phrase="security",
            reference_license_price=5.0,
            integration_costs={firm.firm_id: 1.0 for firm in firms},
            substitute_costs={firm.firm_id: 7.0 for firm in firms},
        ),
    ]
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
                    "build_substitutes": [],
                    "license_sells": [],
                    "license_buys": [],
                    "fulfill_orders": [],
                    "commentary": "",
                },
                '{"stub": true}',
            )
        call_counts["F1"] += 1
        if call_counts["F1"] == 1:
            return (
                {
                    "build_substitutes": [{"capability": "telemetry"}],
                    "license_sells": [],
                    "license_buys": [],
                    "fulfill_orders": ["O1"],
                    "commentary": "start substitute",
                },
                '{"stub": true}',
            )
        return (
            {
                "build_substitutes": [],
                "license_sells": [],
                "license_buys": [],
                "fulfill_orders": ["O1"],
                "commentary": "deliver now",
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
    firms = [Firm(firm_id=f"F{i+1}", system_prompt="stub") for i in range(4)]
    modules = [
        Module(
            tech_id="T1",
            owner="F1",
            quality="High",
            capability="control",
            module_name="Control core",
            description="Control core",
            order_phrase="control",
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
            description="Telemetry stack",
            order_phrase="telemetry",
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
            description="Compliance pack",
            order_phrase="compliance",
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
            description="Security bundle",
            order_phrase="security",
            reference_license_price=5.0,
            integration_costs={firm.firm_id: 1.0 for firm in firms},
            substitute_costs={firm.firm_id: 7.0 for firm in firms},
        ),
    ]
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
                    "build_substitutes": [],
                    "license_sells": [],
                    "license_buys": [],
                    "fulfill_orders": [],
                    "commentary": "",
                },
                '{"stub": true}',
            )
        return (
            {
                "build_substitutes": [{"capability": "T2"}],
                "license_sells": [],
                "license_buys": [],
                "fulfill_orders": ["O1"],
                "commentary": "try to copy rival module by id",
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
    )

    assert result["completed_substitutes_by_firm"]["F1"] == []
    assert result["summary"]["orders_fulfilled"] == 0
