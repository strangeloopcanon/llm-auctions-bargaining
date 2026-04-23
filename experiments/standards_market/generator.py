from __future__ import annotations

import random
from typing import Dict, List

from .models import Firm, JointProject
from .prompts import firm_prompt


DEFAULT_ROUNDS = 3
DEFAULT_DEADLINE_ROUND = 3
FORMATS = ("Alpha", "Beta", "Gamma")
MODULE_NAMES = (
    "sensor array",
    "control board",
    "power unit",
    "communications layer",
)
PROJECT_NAMES = (
    "warehouse automation kit",
    "fleet telemetry platform",
    "cold-chain control system",
    "industrial safety network",
)


def generate_firms(n_firms: int, arm: str) -> List[Firm]:
    if n_firms != 4:
        raise ValueError("Standards benchmark currently expects exactly 4 firms.")

    firms: List[Firm] = []
    for index, module_name in enumerate(MODULE_NAMES, start=1):
        preferred_format = FORMATS[(index - 1) % len(FORMATS)]
        firms.append(
            Firm(
                firm_id=f"F{index}",
                module_name=module_name,
                preferred_format=preferred_format,
                system_prompt=firm_prompt(firm_id=f"F{index}", arm=arm),
            )
        )
    return firms


def generate_project(seed: int, *, deadline_round: int = DEFAULT_DEADLINE_ROUND) -> JointProject:
    rng = random.Random(seed)
    project_name = PROJECT_NAMES[seed % len(PROJECT_NAMES)]
    consortium_bonus_per_firm = round(rng.uniform(22.0, 28.0), 2)
    module_completion_bonus = round(rng.uniform(2.0, 4.0), 2)
    return JointProject(
        project_name=project_name,
        customer_brief=(
            f"A buyer wants a completed {project_name} by round {deadline_round}. "
            "Each firm controls one module. The large payoff only arrives if the whole "
            "system passes final integration before the deadline."
        ),
        deadline_round=deadline_round,
        module_completion_bonus=module_completion_bonus,
        consortium_bonus_per_firm=consortium_bonus_per_firm,
    )


def generate_private_costs(seed: int) -> Dict[str, Dict[str, float]]:
    rng = random.Random(seed)
    cost_map: Dict[str, Dict[str, float]] = {}
    preferred_cycle = {
        "F1": "Alpha",
        "F2": "Beta",
        "F3": "Gamma",
        "F4": rng.choice(FORMATS),
    }
    for firm_id, preferred_format in preferred_cycle.items():
        costs: Dict[str, float] = {}
        for format_name in FORMATS:
            if format_name == preferred_format:
                costs[format_name] = round(rng.uniform(3.0, 4.0), 2)
            else:
                costs[format_name] = round(rng.uniform(5.5, 7.5), 2)
        cost_map[firm_id] = costs
    return cost_map
