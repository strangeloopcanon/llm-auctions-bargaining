from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Firm:
    firm_id: str
    system_prompt: str


@dataclass
class Module:
    tech_id: str
    owner: str
    quality: str
    integration_costs: Dict[str, float]
    internal_costs: Dict[str, float]
    benefits: Dict[str, float]
    needs: Dict[str, int]


@dataclass
class Product:
    product_id: str
    target_firm: str
    required_modules: List[str]
    launch_bonus: float
