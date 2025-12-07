import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from ..agents import call_agent_json, openai_client_from_settings
from ..protocols import Event, event_log_to_jsonl, write_jsonl
from ..settings import ModelSettings
from .allocator import select_features_by_ratio
from .generator import build_department_views, generate_features
from .models import Department, Feature
from .prompts import DEPARTMENT_SYSTEM_PROMPTS


@dataclass
class SimConfig:
    feature_count: int = 20
    quarters: int = 10
    base_budget: float = 25.0
    mu: float = 0.25  # thrift weight
    phi: float = 0.5  # budget responsiveness
    budget_min: float = 5.0
    budget_max: float = 60.0
    capacity_ratio: float = 0.4
    risk_penalty_per_feature: float = 5.0
    engineering_veto_on_risk: bool = True
    outage_base_prob: float = 0.02
    outage_per_risky: float = 0.05
    outage_cost: float = 30.0


def init_departments(base_budget: float) -> List[Department]:
    depts: List[Department] = []
    for name, prompt in DEPARTMENT_SYSTEM_PROMPTS.items():
        depts.append(
            Department(
                name=name,
                system_prompt=prompt,
                base_budget=base_budget,
                budget=base_budget,
            )
        )
    return depts


def build_user_message(dept: Department, views: List[dict], capacity: int, prev_strength: List[dict]) -> dict:
    last_payoff = dept.utility_history[-1] if dept.utility_history else 0.0
    history_slice = dept.utility_history[-5:]
    return {
        "your_department": dept.name,
        "your_current_points": int(dept.budget),
        "engineering_capacity": capacity,
        "previous_quarter_summary": {
            "your_utility_last_quarter": last_payoff,
            "recent_utilities": history_slice,
            "your_points_last_quarter": dept.budget,
            "other_departments_relative_strength": prev_strength,
        },
        "features": [
            {
                "feature_id": v.feature_id,
                "title": v.title,
                "eng_cost": v.eng_cost,
                "department_view": v.department_view,
                "current_total_support_last_quarter": v.current_total_support_last_quarter,
            }
            for v in views
        ],
        "instructions": "Return JSON: {\"commentary\": string, \"bids\": [{\"feature_id\": str, \"points\": int}...]}. Do not exceed your current point budget. Points must be non-negative integers.",
    }


def parse_bids(raw: Dict, budget: float, valid_ids: List[str]) -> Dict[str, int]:
    bids_raw = raw.get("bids") if isinstance(raw, dict) else []
    bids: Dict[str, int] = {}
    total = 0
    for entry in bids_raw or []:
        fid = entry.get("feature_id")
        pts = entry.get("points")
        if fid in valid_ids and isinstance(pts, (int, float)) and pts >= 0:
            bids[fid] = int(pts)
            total += int(pts)
    if total > budget:
        # scale down proportionally
        factor = budget / total if total > 0 else 0
        bids = {fid: max(0, int(pts * factor)) for fid, pts in bids.items()}
    return bids


def planner_optimum(features: List[Feature], capacity: int) -> Tuple[List[str], float]:
    """
    Simple knapsack on average ground-truth value across departments.
    """
    items = []
    for f in features:
        avg_val = sum(f.ground_truth_values.values()) / len(f.ground_truth_values)
        items.append((f, avg_val, f.eng_cost))
    n = len(items)
    dp = [[0.0] * (capacity + 1) for _ in range(n + 1)]
    keep = [[False] * (capacity + 1) for _ in range(n + 1)]
    for i, (f, val, cost) in enumerate(items, start=1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i - 1][w]
            if cost <= w:
                cand = dp[i - 1][w - cost] + val
                if cand > dp[i][w]:
                    dp[i][w] = cand
                    keep[i][w] = True
    w = capacity
    selected: List[str] = []
    for i in range(n, 0, -1):
        if keep[i][w]:
            f = items[i - 1][0]
            selected.append(f.feature_id)
            w -= items[i - 1][2]
    return list(reversed(selected)), dp[n][capacity]


def run_quarter(
    model_settings: ModelSettings,
    departments: List[Department],
    features: List[Feature],
    support_prev: Dict[str, int],
    sim_cfg: SimConfig,
    seed: int,
    dry_run: bool = False,
) -> Tuple[Dict, Dict[str, int], List[Feature]]:
    rng_seed = seed
    dept_views = build_department_views(features, seed=rng_seed)
    capacity = int(sim_cfg.capacity_ratio * sum(f.eng_cost for f in features))
    prev_strength = [{"department": d.name, "budget": d.budget} for d in departments]
    bids_by_dept: Dict[str, Dict[str, int]] = {}
    commentary: Dict[str, str] = {}
    client = None
    if not dry_run and model_settings.provider == "openai":
        client = openai_client_from_settings(model_settings)
    for dept in departments:
        user_msg = build_user_message(
            dept=dept,
            views=dept_views[dept.name],
            capacity=capacity,
            prev_strength=prev_strength,
        )
        schema = {
            "type": "OBJECT",
            "properties": {
                "commentary": {"type": "STRING"},
                "bids": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {"feature_id": {"type": "STRING"}, "points": {"type": "INTEGER"}},
                        "required": ["feature_id", "points"],
                    },
                },
            },
            "required": ["bids"],
        }
        data, raw_text = call_agent_json(
            client=client,
            model_settings=model_settings,
            system_prompt=dept.system_prompt,
            user_prompt=json.dumps(user_msg),
            response_schema=schema,
            dry_run=dry_run,
        )
        bids = parse_bids(data, budget=dept.budget, valid_ids=[f.feature_id for f in features])
        bids_by_dept[dept.name] = bids
        commentary[dept.name] = data.get("commentary", "") if isinstance(data, dict) else ""
    # Aggregate support
    support: Dict[str, int] = {f.feature_id: 0 for f in features}
    for dept_name, bids in bids_by_dept.items():
        for fid, pts in bids.items():
            support[fid] += pts
    # Selection
    selected = select_features_by_ratio(features, support, capacity)
    selected_ids = {f.feature_id for f in selected}
    vetoed: List[str] = []
    risky_flags = {"tech_debt_risk", "security_sensitive", "operational_burden"}
    if sim_cfg.engineering_veto_on_risk:
        eng_bids = bids_by_dept.get("Engineering", {})
        kept = []
        for f in selected:
            if f.risk_flag in risky_flags and eng_bids.get(f.feature_id, 0) <= 0:
                vetoed.append(f.feature_id)
                continue
            kept.append(f)
        selected = kept
        selected_ids = {f.feature_id for f in selected}
    risky_selected = [f for f in selected if f.risk_flag in risky_flags]
    # outage simulation
    import random

    p_outage = sim_cfg.outage_base_prob + sim_cfg.outage_per_risky * len(risky_selected)
    outage = random.random() < p_outage
    # Utilities and budget updates
    utilities: Dict[str, float] = {}
    for dept in departments:
        util_from_features = sum(f.ground_truth_values[dept.name] for f in selected)
        spent = sum(bids_by_dept[dept.name].values())
        unspent = dept.budget - spent
        platform_penalty = sim_cfg.risk_penalty_per_feature * len(risky_selected)
        outage_hit = sim_cfg.outage_cost if outage else 0.0
        U_d = util_from_features + sim_cfg.mu * unspent - platform_penalty - outage_hit
        utilities[dept.name] = U_d
        dept.utility_history.append(U_d)
        new_budget = dept.base_budget + sim_cfg.phi * U_d
        dept.budget = max(sim_cfg.budget_min, min(sim_cfg.budget_max, new_budget))
    # Update support summaries for next round
    for dept in departments:
        for v in dept_views[dept.name]:
            v.current_total_support_last_quarter = support.get(v.feature_id, 0)
    quarter_log = {
        "support": support,
        "selected": list(selected_ids),
        "vetoed": vetoed,
        "risky_selected": [f.feature_id for f in risky_selected],
        "outage": outage,
        "p_outage": p_outage,
        "bids": bids_by_dept,
        "utilities": utilities,
        "budgets": {d.name: d.budget for d in departments},
        "commentary": commentary,
    }
    return quarter_log, support, selected


def run_simulation(
    model_settings: ModelSettings,
    sim_cfg: SimConfig,
    seed: int,
    output_dir: Path,
    dry_run: bool = False,
) -> Dict:
    rng_seed = seed
    features = generate_features(sim_cfg.feature_count, seed=rng_seed + 10)
    departments = init_departments(sim_cfg.base_budget)
    support_prev = {f.feature_id: 0 for f in features}
    capacity = int(sim_cfg.capacity_ratio * sum(f.eng_cost for f in features))
    planner_selection, planner_value = planner_optimum(features, capacity)
    events: List[Event] = []
    summary_quarters: List[Dict] = []
    for q in range(sim_cfg.quarters):
        quarter_log, support_prev, selected = run_quarter(
            model_settings=model_settings,
            departments=departments,
            features=features,
            support_prev=support_prev,
            sim_cfg=sim_cfg,
            seed=rng_seed + q + 100,
            dry_run=dry_run,
        )
        summary_quarters.append({"quarter": q + 1, **quarter_log})
        events.append(Event.now("quarter_result", {"quarter": q + 1, **quarter_log}))
    output_dir.mkdir(parents=True, exist_ok=True)
    event_log_to_jsonl(output_dir / f"firm_market_seed{seed}_events.jsonl", events)
    write_jsonl(
        output_dir / f"firm_market_seed{seed}_summary.jsonl",
        [
            {
                "seed": seed,
                "quarters": sim_cfg.quarters,
                "feature_count": sim_cfg.feature_count,
                "planner_selection": planner_selection,
                "planner_value": planner_value,
                "results": summary_quarters,
            }
        ],
    )
    return {"seed": seed, "results": summary_quarters}
