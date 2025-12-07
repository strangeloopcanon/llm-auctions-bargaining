from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Department:
    name: str
    system_prompt: str
    base_budget: float
    budget: float
    utility_history: List[float] = field(default_factory=list)


@dataclass
class Feature:
    feature_id: str
    title: str
    description: str
    eng_cost: int
    ground_truth_values: Dict[str, float]
    risk_flag: str = "normal"  # e.g., normal, tech_debt_risk, security_sensitive, operational_burden


@dataclass
class DepartmentView:
    feature_id: str
    title: str
    eng_cost: int
    department_view: Dict[str, str]
    current_total_support_last_quarter: int = 0
