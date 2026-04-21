from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Sequence, Set, Tuple

from ..agents import call_agent_json, openai_client_from_settings
from ..protocols import safe_float
from ..settings import ModelSettings
from .models import Firm, Module, Product


def parse_firm_plan(raw: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str], str]:
    build_decisions = raw.get("build_decisions") if isinstance(raw, dict) else []
    license_sells = raw.get("license_sells") if isinstance(raw, dict) else []
    license_buys = raw.get("license_buys") if isinstance(raw, dict) else []
    launch_products = raw.get("launch_products") if isinstance(raw, dict) else []
    commentary = raw.get("commentary", "") if isinstance(raw, dict) else ""

    launch_ids = [str(item).strip() for item in list(launch_products or []) if str(item).strip()]
    return (
        list(build_decisions or []),
        list(license_sells or []),
        list(license_buys or []),
        launch_ids,
        commentary,
    )


def price_hints_for_modules(modules: Sequence[Module]) -> Tuple[Dict[str, float], Dict[str, float]]:
    asks_hint: Dict[str, float] = {}
    bids_hint: Dict[str, float] = {}
    for module in modules:
        other_benefits = [
            benefit for firm_id, benefit in module.benefits.items() if firm_id != module.owner
        ]
        asks_hint[module.tech_id] = round(
            max(0.0, sum(other_benefits) / len(other_benefits)) if other_benefits else 0.0,
            2,
        )
        bids_hint[module.tech_id] = round(
            max(
                0.0,
                max(
                    (
                        module.benefits[firm_id] - module.integration_costs[firm_id]
                        for firm_id in module.benefits
                        if firm_id != module.owner
                    ),
                    default=0.0,
                ),
            ),
            2,
        )
    return asks_hint, bids_hint


def positive_build_decision(value: Any) -> bool:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not token:
        return False
    negative_markers = ("skip", "do_not_build", "dont_build", "no_build", "hold")
    if any(marker in token for marker in negative_markers):
        return False
    return "build" in token


def product_state_snapshot(
    *,
    products: Sequence[Product],
    holdings_by_firm: Dict[str, Set[str]],
    attempted_launches: Set[str],
    launched_products: Set[str],
) -> Dict[str, Dict[str, Any]]:
    snapshot: Dict[str, Dict[str, Any]] = {}
    for product in products:
        held_modules = sorted(
            tech_id
            for tech_id in product.required_modules
            if tech_id in holdings_by_firm[product.target_firm]
        )
        missing_modules = sorted(
            tech_id
            for tech_id in product.required_modules
            if tech_id not in holdings_by_firm[product.target_firm]
        )
        started = product.product_id in attempted_launches or bool(held_modules)
        snapshot[product.product_id] = {
            "target_firm": product.target_firm,
            "required_modules": list(product.required_modules),
            "acquired_modules": held_modules,
            "missing_modules": missing_modules,
            "launch_bonus": product.launch_bonus,
            "launch_ready": not missing_modules,
            "started": started,
            "launched": product.product_id in launched_products,
        }
    return snapshot


def started_not_finished_products(product_state: Dict[str, Dict[str, Any]]) -> List[str]:
    return sorted(
        product_id
        for product_id, state in product_state.items()
        if state.get("started") and not state.get("launched")
    )


def evaluate_launch_attempts(
    *,
    products_by_id: Dict[str, Product],
    holdings_by_firm: Dict[str, Set[str]],
    launch_attempts_by_firm: Dict[str, List[str]],
    launched_products: Set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
    successful_launches: List[Dict[str, Any]] = []
    failed_launches: List[Dict[str, Any]] = []
    launch_bonus = 0.0

    for firm_id, product_ids in launch_attempts_by_firm.items():
        for product_id in product_ids:
            product = products_by_id.get(product_id)
            if product is None:
                failed_launches.append(
                    {"firm_id": firm_id, "product_id": product_id, "reason": "unknown_product"}
                )
                continue
            if product.target_firm != firm_id:
                failed_launches.append(
                    {
                        "firm_id": firm_id,
                        "product_id": product_id,
                        "reason": "wrong_firm",
                        "target_firm": product.target_firm,
                    }
                )
                continue
            if product_id in launched_products:
                failed_launches.append(
                    {"firm_id": firm_id, "product_id": product_id, "reason": "already_launched"}
                )
                continue
            missing_modules = [
                tech_id
                for tech_id in product.required_modules
                if tech_id not in holdings_by_firm[firm_id]
            ]
            if missing_modules:
                failed_launches.append(
                    {
                        "firm_id": firm_id,
                        "product_id": product_id,
                        "reason": "missing_modules",
                        "missing_modules": missing_modules,
                    }
                )
                continue

            launched_products.add(product_id)
            launch_bonus += product.launch_bonus
            successful_launches.append(
                {
                    "firm_id": firm_id,
                    "product_id": product_id,
                    "launch_bonus": product.launch_bonus,
                }
            )

    return successful_launches, failed_launches, launch_bonus


def build_summary(
    *,
    products: Sequence[Product],
    product_state: Dict[str, Dict[str, Any]],
    total_deal_volume: int,
    total_internal_builds: int,
    total_welfare: float,
    launch_bonus_captured: float,
) -> Dict[str, Any]:
    total_products = len(products)
    launches = sum(1 for state in product_state.values() if state.get("launched"))
    started_not_finished = started_not_finished_products(product_state)
    return {
        "total_products": total_products,
        "launches": launches,
        "launch_rate": round(launches / total_products, 4) if total_products else 0.0,
        "partial_bundle_rate": round(len(started_not_finished) / total_products, 4)
        if total_products
        else 0.0,
        "started_not_finished_count": len(started_not_finished),
        "started_not_finished_products": started_not_finished,
        "deal_volume": total_deal_volume,
        "internal_builds": total_internal_builds,
        "welfare": round(total_welfare, 4),
        "launch_bonus_captured": round(launch_bonus_captured, 4),
    }


def _module_index(modules: Sequence[Module]) -> Dict[str, Module]:
    return {module.tech_id: module for module in modules}


def _product_index(products: Sequence[Product]) -> Dict[str, Product]:
    return {product.product_id: product for product in products}


def _products_by_firm(products: Sequence[Product]) -> Dict[str, List[Product]]:
    grouped: Dict[str, List[Product]] = {}
    for product in products:
        grouped.setdefault(product.target_firm, []).append(product)
    return grouped


def _initial_holdings(modules: Sequence[Module]) -> Dict[str, Set[str]]:
    holdings: Dict[str, Set[str]] = {}
    for module in modules:
        holdings.setdefault(module.owner, set()).add(module.tech_id)
    return holdings


def _firm_private_input(
    *,
    firm: Firm,
    modules: Sequence[Module],
    products: Sequence[Product],
    product_state: Dict[str, Dict[str, Any]],
    holdings_by_firm: Dict[str, Set[str]],
    reputation: Dict[str, float],
    last_round_deals: List[Dict[str, Any]],
    history: List[Dict[str, Any]],
    asks_hint: Dict[str, float],
    bids_hint: Dict[str, float],
    score: float,
    rank: int,
    total_firms: int,
) -> Dict[str, Any]:
    modules_view = []
    for module in modules:
        modules_view.append(
            {
                "tech_id": module.tech_id,
                "owner": module.owner,
                "quality": module.quality,
                "you_currently_hold_it": module.tech_id in holdings_by_firm.get(firm.firm_id, set()),
                "need_level": module.needs.get(firm.firm_id, 0),
                "benefit": module.benefits.get(firm.firm_id, 0.0),
                "internal_cost": module.internal_costs.get(firm.firm_id, 0.0),
                "integration_cost": module.integration_costs.get(firm.firm_id, 0.0),
                "suggested_ask": asks_hint.get(module.tech_id, 0.0),
                "suggested_bid": bids_hint.get(module.tech_id, 0.0),
            }
        )

    target_products = []
    for product in products:
        state = product_state[product.product_id]
        target_products.append(
            {
                "product_id": product.product_id,
                "launch_bonus": product.launch_bonus,
                "required_modules": list(product.required_modules),
                "currently_acquired_modules": list(state["acquired_modules"]),
                "missing_modules": list(state["missing_modules"]),
                "launch_ready": state["launch_ready"],
            }
        )

    return {
        "firm_id": firm.firm_id,
        "owned_modules": sorted(holdings_by_firm.get(firm.firm_id, set())),
        "target_products": target_products,
        "module_market": modules_view,
        "transaction_cost_per_deal": 0.5,
        "reputation_score": reputation.get(firm.firm_id, 0.0),
        "last_round_deals": last_round_deals,
        "recent_history": history[-3:],
        "your_score_last_round": score,
        "your_rank_last_round": rank,
        "total_firms": total_firms,
    }


def _fallback_license_sell(
    *,
    firm_id: str,
    modules: Sequence[Module],
    asks_hint: Dict[str, float],
) -> List[Dict[str, Any]]:
    owned_modules = [module for module in modules if module.owner == firm_id]
    if not owned_modules:
        return []
    target = max(owned_modules, key=lambda module: asks_hint.get(module.tech_id, 0.0))
    return [{"tech_id": target.tech_id, "min_price": max(0.0, asks_hint.get(target.tech_id, 1.0))}]


def _fallback_license_buy(
    *,
    firm_id: str,
    modules_by_id: Dict[str, Module],
    target_products: Sequence[Product],
    holdings_by_firm: Dict[str, Set[str]],
    bids_hint: Dict[str, float],
) -> List[Dict[str, Any]]:
    missing_required_modules: List[str] = []
    current_holdings = holdings_by_firm.get(firm_id, set())
    for product in target_products:
        for tech_id in product.required_modules:
            if tech_id not in current_holdings:
                missing_required_modules.append(tech_id)
    if missing_required_modules:
        tech_id = max(missing_required_modules, key=lambda item: bids_hint.get(item, 0.0))
        return [{"tech_id": tech_id, "max_price": max(0.0, bids_hint.get(tech_id, 1.0))}]

    candidates = [
        module
        for module in modules_by_id.values()
        if module.owner != firm_id and module.tech_id not in current_holdings and module.needs.get(firm_id, 0) > 0
    ]
    if not candidates:
        return []
    target = max(candidates, key=lambda module: bids_hint.get(module.tech_id, 0.0))
    return [{"tech_id": target.tech_id, "max_price": max(0.0, bids_hint.get(target.tech_id, 1.0))}]


def _deal_failure(
    *,
    seed: int,
    round_index: int,
    tech_id: str,
    seller_id: str,
    buyer_id: str,
    quality: str,
) -> bool:
    failure_probability = {"High": 0.05, "Medium": 0.2, "Low": 0.5}.get(quality, 0.2)
    draw = random.Random(f"{seed}:{round_index}:{tech_id}:{seller_id}:{buyer_id}").random()
    return draw < failure_probability


def run_completion_market(
    model_settings: ModelSettings,
    firms: Sequence[Firm],
    modules: Sequence[Module],
    products: Sequence[Product],
    *,
    dry_run: bool = False,
    rounds: int = 2,
    seed: int = 0,
) -> Dict[str, Any]:
    client = None
    if not dry_run and model_settings.provider == "openai":
        client = openai_client_from_settings(model_settings)

    schema = {
        "type": "OBJECT",
        "properties": {
            "build_decisions": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "tech_id": {"type": "STRING"},
                        "decision": {"type": "STRING"},
                    },
                    "required": ["tech_id", "decision"],
                },
            },
            "license_sells": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "tech_id": {"type": "STRING"},
                        "min_price": {"type": "NUMBER"},
                    },
                    "required": ["tech_id", "min_price"],
                },
            },
            "license_buys": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "tech_id": {"type": "STRING"},
                        "max_price": {"type": "NUMBER"},
                    },
                    "required": ["tech_id", "max_price"],
                },
            },
            "launch_products": {
                "type": "ARRAY",
                "items": {"type": "STRING"},
            },
            "commentary": {"type": "STRING"},
        },
    }

    modules_by_id = _module_index(modules)
    products_by_id = _product_index(products)
    products_by_firm = _products_by_firm(products)
    holdings_by_firm = _initial_holdings(modules)
    built_modules_by_firm: Dict[str, Set[str]] = {firm.firm_id: set() for firm in firms}
    licensed_modules_by_firm: Dict[str, Set[str]] = {firm.firm_id: set() for firm in firms}
    reputation = {firm.firm_id: 0.0 for firm in firms}
    history_by_firm: Dict[str, List[Dict[str, Any]]] = {firm.firm_id: [] for firm in firms}
    last_round_deals: List[Dict[str, Any]] = []
    launched_products: Set[str] = set()
    total_welfare = 0.0
    launch_bonus_captured = 0.0
    total_internal_builds = 0
    total_deal_volume = 0
    round_logs: List[Dict[str, Any]] = []
    profits_prev: Dict[str, float] = {firm.firm_id: 0.0 for firm in firms}

    for round_index in range(1, rounds + 1):
        asks_hint, bids_hint = price_hints_for_modules(modules)
        product_state_before = product_state_snapshot(
            products=products,
            holdings_by_firm=holdings_by_firm,
            attempted_launches=set(),
            launched_products=launched_products,
        )
        sorted_firms = sorted(
            firms,
            key=lambda firm: profits_prev.get(firm.firm_id, 0.0),
            reverse=True,
        )
        ranks = {firm.firm_id: index + 1 for index, firm in enumerate(sorted_firms)}

        plans: Dict[str, Dict[str, Any]] = {}
        usage_by_firm: Dict[str, Dict[str, int]] = {}
        attempted_launches: Set[str] = set()
        round_profits: Dict[str, float] = {firm.firm_id: 0.0 for firm in firms}
        successful_builds: List[Dict[str, Any]] = []

        for firm in firms:
            metadata: Dict[str, Any] = {}
            target_products = products_by_firm.get(firm.firm_id, [])
            user_payload = _firm_private_input(
                firm=firm,
                modules=modules,
                products=target_products,
                product_state=product_state_before,
                holdings_by_firm=holdings_by_firm,
                reputation=reputation,
                last_round_deals=last_round_deals,
                history=history_by_firm[firm.firm_id],
                asks_hint=asks_hint,
                bids_hint=bids_hint,
                score=profits_prev.get(firm.firm_id, 0.0),
                rank=ranks.get(firm.firm_id, len(firms)),
                total_firms=len(firms),
            )
            data, raw_text = call_agent_json(
                client=client,
                model_settings=model_settings,
                system_prompt=firm.system_prompt,
                user_prompt=json.dumps(user_payload),
                response_schema=schema,
                dry_run=dry_run,
                metadata_sink=metadata,
            )
            build_decisions, license_sells, license_buys, launch_products, commentary = parse_firm_plan(
                data
            )
            if not license_sells:
                license_sells = _fallback_license_sell(
                    firm_id=firm.firm_id,
                    modules=modules,
                    asks_hint=asks_hint,
                )
            if not license_buys:
                license_buys = _fallback_license_buy(
                    firm_id=firm.firm_id,
                    modules_by_id=modules_by_id,
                    target_products=target_products,
                    holdings_by_firm=holdings_by_firm,
                    bids_hint=bids_hint,
                )

            plans[firm.firm_id] = {
                "build_decisions": build_decisions,
                "license_sells": license_sells,
                "license_buys": license_buys,
                "launch_products": launch_products,
                "commentary": commentary,
                "raw_text": raw_text,
            }
            if metadata.get("usage"):
                usage_by_firm[firm.firm_id] = metadata["usage"]

        for firm in firms:
            firm_id = firm.firm_id
            for entry in plans[firm_id]["build_decisions"]:
                tech_id = str(entry.get("tech_id") or "").strip()
                if not tech_id or tech_id not in modules_by_id:
                    continue
                if tech_id in holdings_by_firm[firm_id]:
                    continue
                if not positive_build_decision(entry.get("decision")):
                    continue
                module = modules_by_id[tech_id]
                holdings_by_firm.setdefault(firm_id, set()).add(tech_id)
                built_modules_by_firm[firm_id].add(tech_id)
                build_profit = module.benefits[firm_id] - module.internal_costs[firm_id]
                round_profits[firm_id] += build_profit
                total_welfare += build_profit
                total_internal_builds += 1
                successful_builds.append(
                    {
                        "firm_id": firm_id,
                        "tech_id": tech_id,
                        "benefit": module.benefits[firm_id],
                        "internal_cost": module.internal_costs[firm_id],
                        "profit": round(build_profit, 4),
                    }
                )

        successful_deals: List[Dict[str, Any]] = []
        failed_deals: List[Dict[str, Any]] = []
        for module in modules:
            asks: List[Tuple[str, float]] = []
            bids: List[Tuple[str, float]] = []
            for firm_id, plan in plans.items():
                if firm_id == module.owner:
                    for ask in plan["license_sells"]:
                        if str(ask.get("tech_id") or "").strip() != module.tech_id:
                            continue
                        asks.append((firm_id, safe_float(ask.get("min_price"), 0.0) or 0.0))
                for bid in plan["license_buys"]:
                    if str(bid.get("tech_id") or "").strip() != module.tech_id:
                        continue
                    if firm_id == module.owner or module.tech_id in holdings_by_firm.get(firm_id, set()):
                        continue
                    bids.append((firm_id, safe_float(bid.get("max_price"), 0.0) or 0.0))
            if not asks or not bids:
                continue

            seller_id, ask_price = min(asks, key=lambda item: item[1])
            buyer_id, bid_price = max(bids, key=lambda item: item[1])
            if bid_price < ask_price:
                continue

            clearing_price = round((bid_price + ask_price) / 2.0, 4)
            if _deal_failure(
                seed=seed,
                round_index=round_index,
                tech_id=module.tech_id,
                seller_id=seller_id,
                buyer_id=buyer_id,
                quality=module.quality,
            ):
                reputation[seller_id] -= 2.0
                reputation[buyer_id] -= 1.0
                failed_deals.append(
                    {
                        "tech_id": module.tech_id,
                        "seller": seller_id,
                        "buyer": buyer_id,
                        "price": clearing_price,
                    }
                )
                continue

            holdings_by_firm.setdefault(buyer_id, set()).add(module.tech_id)
            licensed_modules_by_firm[buyer_id].add(module.tech_id)
            reputation[seller_id] += 1.0
            reputation[buyer_id] += 0.5

            buyer_profit = module.benefits[buyer_id] - module.integration_costs[buyer_id] - clearing_price - 0.25
            seller_profit = clearing_price - 0.25
            round_profits[buyer_id] += buyer_profit
            round_profits[seller_id] += seller_profit
            total_welfare += module.benefits[buyer_id] - module.integration_costs[buyer_id] - 0.5
            total_deal_volume += 1
            successful_deals.append(
                {
                    "tech_id": module.tech_id,
                    "seller": seller_id,
                    "buyer": buyer_id,
                    "price": clearing_price,
                    "buyer_profit": round(buyer_profit, 4),
                    "seller_profit": round(seller_profit, 4),
                }
            )

        launch_attempts_by_firm = {
            firm_id: list(plan["launch_products"]) for firm_id, plan in plans.items() if plan["launch_products"]
        }
        attempted_launches = {
            product_id for product_ids in launch_attempts_by_firm.values() for product_id in product_ids
        }
        successful_launches, failed_launches, round_launch_bonus = evaluate_launch_attempts(
            products_by_id=products_by_id,
            holdings_by_firm=holdings_by_firm,
            launch_attempts_by_firm=launch_attempts_by_firm,
            launched_products=launched_products,
        )
        for launch in successful_launches:
            round_profits[launch["firm_id"]] += launch["launch_bonus"]
        total_welfare += round_launch_bonus
        launch_bonus_captured += round_launch_bonus

        product_state_after = product_state_snapshot(
            products=products,
            holdings_by_firm=holdings_by_firm,
            attempted_launches=attempted_launches,
            launched_products=launched_products,
        )
        started_not_finished = started_not_finished_products(product_state_after)

        round_usage = None
        if usage_by_firm:
            round_usage = {
                "calls": sum(item.get("calls", 0) for item in usage_by_firm.values()),
                "input_tokens": sum(item.get("input_tokens", 0) for item in usage_by_firm.values()),
                "output_tokens": sum(item.get("output_tokens", 0) for item in usage_by_firm.values()),
            }

        for firm in firms:
            history_by_firm[firm.firm_id].append(
                {
                    "round": round_index,
                    "profit": round(round_profits[firm.firm_id], 4),
                    "reputation": round(reputation[firm.firm_id], 4),
                    "held_modules": sorted(holdings_by_firm.get(firm.firm_id, set())),
                    "successful_deals": [
                        deal
                        for deal in successful_deals
                        if deal["buyer"] == firm.firm_id or deal["seller"] == firm.firm_id
                    ],
                    "successful_launches": [
                        launch for launch in successful_launches if launch["firm_id"] == firm.firm_id
                    ],
                }
            )

        profits_prev = dict(round_profits)
        round_logs.append(
            {
                "round": round_index,
                "plans": {
                    firm_id: {
                        "build_decisions": plan["build_decisions"],
                        "license_sells": plan["license_sells"],
                        "license_buys": plan["license_buys"],
                        "launch_products": plan["launch_products"],
                        "commentary": plan["commentary"],
                    }
                    for firm_id, plan in plans.items()
                },
                "successful_builds": successful_builds,
                "successful_deals": successful_deals,
                "failed_deals": failed_deals,
                "launch_attempts": launch_attempts_by_firm,
                "successful_launches": successful_launches,
                "failed_launches": failed_launches,
                "product_state": product_state_after,
                "started_not_finished": started_not_finished,
                "holdings_by_firm": {
                    firm_id: sorted(holdings) for firm_id, holdings in holdings_by_firm.items()
                },
                "reputation": {firm_id: round(value, 4) for firm_id, value in reputation.items()},
                "profits": {firm_id: round(value, 4) for firm_id, value in round_profits.items()},
                "welfare": round(total_welfare, 4),
                "codex_usage": round_usage,
            }
        )
        last_round_deals = successful_deals

    final_product_state = product_state_snapshot(
        products=products,
        holdings_by_firm=holdings_by_firm,
        attempted_launches=set(),
        launched_products=launched_products,
    )
    summary = build_summary(
        products=products,
        product_state=final_product_state,
        total_deal_volume=total_deal_volume,
        total_internal_builds=total_internal_builds,
        total_welfare=total_welfare,
        launch_bonus_captured=launch_bonus_captured,
    )

    return {
        "summary": summary,
        "round_logs": round_logs,
        "products": [
            {
                "product_id": product.product_id,
                "target_firm": product.target_firm,
                "required_modules": list(product.required_modules),
                "launch_bonus": product.launch_bonus,
            }
            for product in products
        ],
        "modules": [
            {"tech_id": module.tech_id, "owner": module.owner, "quality": module.quality}
            for module in modules
        ],
        "built_modules_by_firm": {
            firm_id: sorted(items) for firm_id, items in built_modules_by_firm.items()
        },
        "licensed_modules_by_firm": {
            firm_id: sorted(items) for firm_id, items in licensed_modules_by_firm.items()
        },
    }
