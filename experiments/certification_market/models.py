from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Firm:
    firm_id: str
    system_prompt: str


@dataclass
class CustomerOrder:
    order_id: str
    firm_id: str
    product_name: str
    customer_brief: str
    delivery_value: float
    build_cost: float
    deadline_round: int


@dataclass
class BoardNotice:
    notice_id: str
    author: str
    headline: str
    body: str
    round_posted: int


@dataclass
class BoardReply:
    reply_id: str
    notice_id: str
    author: str
    body: str
    round_posted: int


@dataclass
class SlotClaim:
    claim_id: str
    resource: str
    notice_id: str
    claimant: str
    order_id: str
    target_round: int
    round_posted: int
    accepted: bool
    used: bool = False
