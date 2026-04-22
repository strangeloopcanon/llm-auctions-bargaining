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
    capability: str
    module_name: str
    description: str
    order_phrase: str
    reference_license_price: float
    integration_costs: Dict[str, float]
    substitute_costs: Dict[str, float]


@dataclass
class CustomerOrder:
    order_id: str
    target_firm: str
    customer_brief: str
    delivery_value: float
    deadline_round: int
    required_capabilities: List[str]


@dataclass
class InternalProject:
    firm_id: str
    capability: str
    source_module_id: str
    start_round: int
    ready_round: int
    cost: float


@dataclass
class BoardNotice:
    notice_id: str
    author: str
    headline: str
    body: str
    cash_terms: str
    round_posted: int


@dataclass
class BoardReply:
    reply_id: str
    notice_id: str
    author: str
    body: str
    cash_offer: float
    round_posted: int
