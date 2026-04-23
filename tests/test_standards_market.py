from __future__ import annotations

import json

import pytest

from experiments.settings import ModelSettings
from experiments.standards_market.generator import FORMATS, generate_firms, generate_private_costs, generate_project
from experiments.standards_market.market import _private_input, run_standards_market
from experiments.standards_market.models import Firm, JointProject


def _sample_firms() -> list[Firm]:
    return [
        Firm(firm_id="F1", module_name="sensor array", preferred_format="Alpha", system_prompt="stub"),
        Firm(firm_id="F2", module_name="control board", preferred_format="Beta", system_prompt="stub"),
        Firm(firm_id="F3", module_name="power unit", preferred_format="Gamma", system_prompt="stub"),
        Firm(firm_id="F4", module_name="communications layer", preferred_format="Alpha", system_prompt="stub"),
    ]


def _sample_project() -> JointProject:
    return JointProject(
        project_name="warehouse automation kit",
        customer_brief="Four modules must integrate by round 3.",
        deadline_round=3,
        module_completion_bonus=3.0,
        consortium_bonus_per_firm=24.0,
    )


def _sample_private_costs() -> dict[str, dict[str, float]]:
    return {
        "F1": {"Alpha": 3.0, "Beta": 6.0, "Gamma": 7.0},
        "F2": {"Alpha": 6.0, "Beta": 3.0, "Gamma": 7.0},
        "F3": {"Alpha": 6.5, "Beta": 7.0, "Gamma": 3.5},
        "F4": {"Alpha": 3.5, "Beta": 6.0, "Gamma": 7.0},
    }


def test_generate_private_costs_is_deterministic() -> None:
    assert generate_private_costs(77) == generate_private_costs(77)


def test_private_input_exposes_formats_but_not_exact_hidden_rule() -> None:
    payload = _private_input(
        firm=_sample_firms()[0],
        project=_sample_project(),
        private_costs=_sample_private_costs()["F1"],
        visible_memos=[],
        build_records=[],
        score=0.0,
        rank=1,
        total_firms=4,
        current_round=1,
        total_rounds=3,
    )

    assert payload["your_module"]["available_formats"] == list(FORMATS)
    assert "same interface format" not in json.dumps(payload).lower()


def test_public_memos_are_visible_next_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    firms = _sample_firms()
    project = _sample_project()
    private_costs = _sample_private_costs()

    def fake_call_agent_json(*args, **kwargs):  # type: ignore[override]
        payload = json.loads(kwargs["user_prompt"])
        if payload["firm_id"] == "F1" and payload["current_round"] == 1:
            return (
                {
                    "public_memo": {
                        "subject": "Format note",
                        "body": "We should align before building.",
                    },
                    "start_build": {},
                    "commentary": "",
                },
                '{"stub": true}',
            )
        return (
            {
                "public_memo": {},
                "start_build": {},
                "commentary": "",
            },
            '{"stub": true}',
        )

    monkeypatch.setattr("experiments.standards_market.market.call_agent_json", fake_call_agent_json)

    result = run_standards_market(
        model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        firms=firms,
        project=project,
        private_costs_by_firm=private_costs,
        rounds=2,
    )

    assert result["round_logs"][0]["visible_memos"] == []
    assert result["round_logs"][1]["visible_memos"][0]["author"] == "F1"


def test_build_completes_next_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    firms = _sample_firms()
    project = _sample_project()
    private_costs = _sample_private_costs()

    def fake_call_agent_json(*args, **kwargs):  # type: ignore[override]
        payload = json.loads(kwargs["user_prompt"])
        if payload["firm_id"] == "F1" and payload["current_round"] == 1:
            return (
                {
                    "public_memo": {},
                    "start_build": {"format_choice": "Alpha"},
                    "commentary": "",
                },
                '{"stub": true}',
            )
        return (
            {
                "public_memo": {},
                "start_build": {},
                "commentary": "",
            },
            '{"stub": true}',
        )

    monkeypatch.setattr("experiments.standards_market.market.call_agent_json", fake_call_agent_json)

    result = run_standards_market(
        model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        firms=firms,
        project=project,
        private_costs_by_firm=private_costs,
        rounds=2,
    )

    assert result["round_logs"][0]["completion_events"] == []
    assert result["round_logs"][1]["completion_events"][0]["firm_id"] == "F1"


def test_matching_formats_deliver_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    firms = _sample_firms()
    project = _sample_project()
    private_costs = _sample_private_costs()

    def fake_call_agent_json(*args, **kwargs):  # type: ignore[override]
        payload = json.loads(kwargs["user_prompt"])
        if payload["current_round"] == 1:
            return (
                {
                    "public_memo": {
                        "subject": "Compatibility",
                        "body": "Use Alpha for integration.",
                    },
                    "start_build": {"format_choice": "Alpha"},
                    "commentary": "",
                },
                '{"stub": true}',
            )
        return (
            {
                "public_memo": {},
                "start_build": {},
                "commentary": "",
            },
            '{"stub": true}',
        )

    monkeypatch.setattr("experiments.standards_market.market.call_agent_json", fake_call_agent_json)

    result = run_standards_market(
        model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        firms=firms,
        project=project,
        private_costs_by_firm=private_costs,
        rounds=3,
    )

    assert result["summary"]["modules_completed"] == 4
    assert result["summary"]["consortium_delivery_success"] == 1


def test_mismatched_formats_create_partial_progress_without_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    firms = _sample_firms()
    project = _sample_project()
    private_costs = _sample_private_costs()

    format_by_firm = {"F1": "Alpha", "F2": "Beta", "F3": "Gamma", "F4": "Alpha"}

    def fake_call_agent_json(*args, **kwargs):  # type: ignore[override]
        payload = json.loads(kwargs["user_prompt"])
        if payload["current_round"] == 1:
            return (
                {
                    "public_memo": {},
                    "start_build": {"format_choice": format_by_firm[payload["firm_id"]]},
                    "commentary": "",
                },
                '{"stub": true}',
            )
        return (
            {
                "public_memo": {},
                "start_build": {},
                "commentary": "",
            },
            '{"stub": true}',
        )

    monkeypatch.setattr("experiments.standards_market.market.call_agent_json", fake_call_agent_json)

    result = run_standards_market(
        model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        firms=firms,
        project=project,
        private_costs_by_firm=private_costs,
        rounds=3,
    )

    assert result["summary"]["modules_completed"] == 4
    assert result["summary"]["consortium_delivery_success"] == 0
    assert result["summary"]["partial_progress_without_delivery"] == 1


def test_dry_run_keeps_benchmark_inert() -> None:
    firms = generate_firms(4, arm="baseline")
    project = generate_project(seed=14)
    private_costs = generate_private_costs(seed=14)

    result = run_standards_market(
        model_settings=ModelSettings(provider="codex", model="gpt-5.4"),
        firms=firms,
        project=project,
        private_costs_by_firm=private_costs,
        dry_run=True,
        rounds=3,
    )

    assert result["summary"]["modules_completed"] == 0
    assert result["summary"]["consortium_delivery_success"] == 0
