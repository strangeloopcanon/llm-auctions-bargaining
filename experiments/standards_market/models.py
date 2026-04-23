from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Firm:
    firm_id: str
    module_name: str
    preferred_format: str
    system_prompt: str


@dataclass
class JointProject:
    project_name: str
    customer_brief: str
    deadline_round: int
    module_completion_bonus: float
    consortium_bonus_per_firm: float


@dataclass
class PublicMemo:
    memo_id: str
    author: str
    subject: str
    body: str
    round_posted: int


@dataclass
class BuildRecord:
    firm_id: str
    module_name: str
    format_choice: str
    build_cost: float
    round_started: int
    round_completed: int
