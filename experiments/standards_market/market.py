from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

from ..agents import call_agent_json, openai_client_from_settings
from ..settings import ModelSettings
from .generator import FORMATS
from .models import BuildRecord, Firm, JointProject, PublicMemo


STANDARD_KEYWORDS = (
    "align",
    "common",
    "compatible",
    "convention",
    "format",
    "interface",
    "same",
    "standard",
)


def parse_firm_plan(raw: Dict[str, Any]) -> Tuple[Dict[str, str] | None, str]:
    if not isinstance(raw, dict):
        return None, ""

    memo = raw.get("public_memo")
    parsed_memo: Dict[str, str] | None = None
    if isinstance(memo, dict):
        subject = str(memo.get("subject") or "").strip()
        body = str(memo.get("body") or "").strip()
        if subject or body:
            parsed_memo = {"subject": subject, "body": body}

    start_build = raw.get("start_build")
    parsed_build: Dict[str, str] | None = None
    if isinstance(start_build, dict):
        format_choice = str(start_build.get("format_choice") or "").strip()
        if format_choice:
            parsed_build = {"format_choice": format_choice}

    commentary = str(raw.get("commentary") or "").strip()
    merged = parsed_build
    if parsed_memo is not None:
        if merged is None:
            merged = {}
        merged["memo_subject"] = parsed_memo["subject"]
        merged["memo_body"] = parsed_memo["body"]
    return merged, commentary


def _memo_to_dict(memo: PublicMemo) -> Dict[str, Any]:
    return {
        "memo_id": memo.memo_id,
        "author": memo.author,
        "subject": memo.subject,
        "body": memo.body,
        "round_posted": memo.round_posted,
    }


def _build_to_dict(build: BuildRecord) -> Dict[str, Any]:
    return {
        "firm_id": build.firm_id,
        "module_name": build.module_name,
        "format_choice": build.format_choice,
        "build_cost": build.build_cost,
        "round_started": build.round_started,
        "round_completed": build.round_completed,
    }


def _visible_memos(all_memos: Sequence[PublicMemo], current_round: int) -> List[PublicMemo]:
    return [memo for memo in all_memos if memo.round_posted < current_round]


def _completed_builds(build_records: Sequence[BuildRecord], current_round: int) -> List[BuildRecord]:
    return [build for build in build_records if build.round_completed < current_round]


def _dominant_format(builds: Sequence[BuildRecord]) -> Tuple[str | None, int]:
    counts = Counter(build.format_choice for build in builds)
    if not counts:
        return None, 0
    format_name, count = counts.most_common(1)[0]
    return format_name, count


def _private_input(
    *,
    firm: Firm,
    project: JointProject,
    private_costs: Dict[str, float],
    visible_memos: Sequence[PublicMemo],
    build_records: Sequence[BuildRecord],
    score: float,
    rank: int,
    total_firms: int,
    current_round: int,
    total_rounds: int,
) -> Dict[str, Any]:
    started_build = next((build for build in build_records if build.firm_id == firm.firm_id), None)
    completed_builds = _completed_builds(build_records, current_round)
    dominant_format, dominant_count = _dominant_format(completed_builds)
    return {
        "firm_id": firm.firm_id,
        "current_round": current_round,
        "total_rounds": total_rounds,
        "project": {
            "project_name": project.project_name,
            "customer_brief": project.customer_brief,
            "deadline_round": project.deadline_round,
            "module_completion_bonus": project.module_completion_bonus,
            "consortium_bonus_per_firm": project.consortium_bonus_per_firm,
        },
        "your_module": {
            "module_name": firm.module_name,
            "available_formats": list(FORMATS),
            "private_build_costs": private_costs,
            "preferred_format": firm.preferred_format,
            "build_started": started_build is not None,
            "completed_format": started_build.format_choice if started_build and started_build.round_completed < current_round else None,
        },
        "shared_context": {
            "completed_modules_count": len(completed_builds),
            "dominant_completed_format": dominant_format,
            "dominant_completed_format_count": dominant_count,
            "visible_memos": [_memo_to_dict(memo) for memo in visible_memos],
        },
        "recent_history": [_build_to_dict(build) for build in completed_builds[-4:]],
        "your_score_last_round": round(score, 4),
        "your_rank_last_round": rank,
        "total_firms": total_firms,
    }


def _round_usage(usage_by_firm: Dict[str, Dict[str, int]]) -> Dict[str, int] | None:
    if not usage_by_firm:
        return None
    return {
        "calls": sum(item.get("calls", 0) for item in usage_by_firm.values()),
        "input_tokens": sum(item.get("input_tokens", 0) for item in usage_by_firm.values()),
        "output_tokens": sum(item.get("output_tokens", 0) for item in usage_by_firm.values()),
    }


def _integration_success(build_records: Sequence[BuildRecord], deadline_round: int) -> bool:
    completed = [build for build in build_records if build.round_completed <= deadline_round]
    if len(completed) != 4:
        return False
    return len({build.format_choice for build in completed}) == 1


def build_summary(
    *,
    build_records: Sequence[BuildRecord],
    public_memos: Sequence[PublicMemo],
    project: JointProject,
    total_welfare: float,
    consortium_delivery_success: bool,
) -> Dict[str, Any]:
    completed = [build for build in build_records if build.round_completed <= project.deadline_round]
    dominant_format, dominant_count = _dominant_format(completed)
    format_distribution = Counter(build.format_choice for build in completed)
    standard_memos = [
        memo
        for memo in public_memos
        if any(keyword in f"{memo.subject} {memo.body}".lower() for keyword in STANDARD_KEYWORDS)
    ]
    return {
        "modules_completed": len(completed),
        "module_completion_rate": round(len(completed) / 4, 4),
        "formats_used_count": len(format_distribution),
        "dominant_format": dominant_format,
        "format_convergence_rate": round(dominant_count / len(completed), 4) if completed else 0.0,
        "standard_memo_rate": round(len(standard_memos) / max(1, len(public_memos)), 4),
        "public_memo_count": len(public_memos),
        "consortium_delivery_success": 1 if consortium_delivery_success else 0,
        "consortium_delivery_rate": 1.0 if consortium_delivery_success else 0.0,
        "incompatible_completed_count": len(completed) if completed and not consortium_delivery_success else 0,
        "partial_progress_without_delivery": 1 if completed and not consortium_delivery_success else 0,
        "welfare": round(total_welfare, 4),
        "format_distribution": dict(format_distribution),
    }


def run_standards_market(
    model_settings: ModelSettings,
    firms: Sequence[Firm],
    project: JointProject,
    private_costs_by_firm: Dict[str, Dict[str, float]],
    *,
    dry_run: bool = False,
    rounds: int = 3,
) -> Dict[str, Any]:
    client = None
    if not dry_run and model_settings.provider == "openai":
        client = openai_client_from_settings(model_settings)

    schema = {
        "type": "OBJECT",
        "properties": {
            "public_memo": {
                "type": "OBJECT",
                "properties": {
                    "subject": {"type": "STRING"},
                    "body": {"type": "STRING"},
                },
                "required": ["subject", "body"],
            },
            "start_build": {
                "type": "OBJECT",
                "properties": {
                    "format_choice": {"type": "STRING"},
                },
                "required": ["format_choice"],
            },
            "commentary": {"type": "STRING"},
        },
    }

    scores = {firm.firm_id: 0.0 for firm in firms}
    build_records: List[BuildRecord] = []
    public_memos: List[PublicMemo] = []
    memo_counter = 1
    round_logs: List[Dict[str, Any]] = []

    for current_round in range(1, rounds + 1):
        visible_memos = _visible_memos(public_memos, current_round)
        ranked_scores = sorted(
            ((firm_id, score) for firm_id, score in scores.items()),
            key=lambda item: (-item[1], item[0]),
        )
        ranks = {firm_id: index + 1 for index, (firm_id, _) in enumerate(ranked_scores)}

        usage_by_firm: Dict[str, Dict[str, int]] = {}
        round_memos: List[PublicMemo] = []
        round_builds: List[BuildRecord] = []
        build_failures: List[Dict[str, Any]] = []
        raw_plans: Dict[str, Dict[str, Any]] = {}

        for firm in sorted(firms, key=lambda item: item.firm_id):
            metadata_sink: Dict[str, Any] = {}
            private_input = _private_input(
                firm=firm,
                project=project,
                private_costs=private_costs_by_firm[firm.firm_id],
                visible_memos=visible_memos,
                build_records=build_records,
                score=scores[firm.firm_id],
                rank=ranks.get(firm.firm_id, len(firms)),
                total_firms=len(firms),
                current_round=current_round,
                total_rounds=rounds,
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

            plan, commentary = parse_firm_plan(raw)
            raw_plans[firm.firm_id] = {
                "plan": plan,
                "commentary": commentary,
                "raw_text": raw_text,
            }
            if plan is None:
                continue

            memo_subject = str(plan.get("memo_subject") or "").strip()
            memo_body = str(plan.get("memo_body") or "").strip()
            if memo_subject or memo_body:
                round_memos.append(
                    PublicMemo(
                        memo_id=f"M{memo_counter}",
                        author=firm.firm_id,
                        subject=memo_subject[:140],
                        body=memo_body[:280],
                        round_posted=current_round,
                    )
                )
                memo_counter += 1

            format_choice = str(plan.get("format_choice") or "").strip()
            if not format_choice:
                continue

            if format_choice not in FORMATS:
                build_failures.append(
                    {
                        "firm_id": firm.firm_id,
                        "reason": "unknown_format",
                        "format_choice": format_choice,
                    }
                )
                continue

            existing_build = next((build for build in build_records if build.firm_id == firm.firm_id), None)
            if existing_build is not None:
                build_failures.append(
                    {
                        "firm_id": firm.firm_id,
                        "reason": "build_already_started",
                        "format_choice": format_choice,
                    }
                )
                continue

            build_cost = private_costs_by_firm[firm.firm_id][format_choice]
            build_record = BuildRecord(
                firm_id=firm.firm_id,
                module_name=firm.module_name,
                format_choice=format_choice,
                build_cost=build_cost,
                round_started=current_round,
                round_completed=current_round + 1,
            )
            build_records.append(build_record)
            round_builds.append(build_record)
            scores[firm.firm_id] -= build_cost

        public_memos.extend(round_memos)

        completion_events: List[Dict[str, Any]] = []
        for build in build_records:
            if build.round_completed != current_round:
                continue
            scores[build.firm_id] += project.module_completion_bonus
            completion_events.append(
                {
                    "firm_id": build.firm_id,
                    "module_name": build.module_name,
                    "format_choice": build.format_choice,
                    "module_completion_bonus": project.module_completion_bonus,
                }
            )

        consortium_delivery_success = False
        consortium_delivery_events: List[Dict[str, Any]] = []
        if current_round == project.deadline_round and _integration_success(build_records, project.deadline_round):
            consortium_delivery_success = True
            for firm in firms:
                scores[firm.firm_id] += project.consortium_bonus_per_firm
                consortium_delivery_events.append(
                    {
                        "firm_id": firm.firm_id,
                        "consortium_bonus_per_firm": project.consortium_bonus_per_firm,
                    }
                )

        round_logs.append(
            {
                "round": current_round,
                "visible_memos": [_memo_to_dict(memo) for memo in visible_memos],
                "round_memos": [_memo_to_dict(memo) for memo in round_memos],
                "round_builds": [_build_to_dict(build) for build in round_builds],
                "completion_events": completion_events,
                "build_failures": build_failures,
                "consortium_delivery_success": consortium_delivery_success,
                "consortium_delivery_events": consortium_delivery_events,
                "firm_plans": raw_plans,
                "scores": {firm_id: round(score, 4) for firm_id, score in scores.items()},
                "usage": _round_usage(usage_by_firm),
            }
        )

    total_welfare = sum(scores.values())
    consortium_delivery_success = _integration_success(build_records, project.deadline_round)
    summary = build_summary(
        build_records=build_records,
        public_memos=public_memos,
        project=project,
        total_welfare=total_welfare,
        consortium_delivery_success=consortium_delivery_success,
    )

    return {
        "summary": summary,
        "round_logs": round_logs,
        "project": {
            "project_name": project.project_name,
            "customer_brief": project.customer_brief,
            "deadline_round": project.deadline_round,
            "module_completion_bonus": project.module_completion_bonus,
            "consortium_bonus_per_firm": project.consortium_bonus_per_firm,
        },
        "firms": [
            {
                "firm_id": firm.firm_id,
                "module_name": firm.module_name,
                "preferred_format": firm.preferred_format,
                "private_build_costs": private_costs_by_firm[firm.firm_id],
            }
            for firm in firms
        ],
        "build_records": [_build_to_dict(build) for build in build_records],
        "public_memos": [_memo_to_dict(memo) for memo in public_memos],
    }
