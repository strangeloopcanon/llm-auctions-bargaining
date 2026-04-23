from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Firm:
    firm_id: str
    role: str
    system_prompt: str


@dataclass
class InventoryLot:
    lot_id: str
    seller: str
    quality: str
    capability: str
    lot_name: str
    description: str
    order_phrase: str
    reservation_price: float


@dataclass
class CustomerOrder:
    order_id: str
    buyer: str
    customer_brief: str
    delivery_value: float
    deadline_round: int
    required_capability: str


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
