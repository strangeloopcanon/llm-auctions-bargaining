from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Firm:
    firm_id: str
    system_prompt: str


@dataclass
class Module:
    tech_id: str
    owner: str  # firm_id that owns patent
    quality: str
    integration_costs: Dict[str, float]  # per firm
    internal_costs: Dict[str, float]  # per firm if built internally
    benefits: Dict[str, float]  # per firm if used
    needs: Dict[str, int]  # 0/1/2 need level per firm
