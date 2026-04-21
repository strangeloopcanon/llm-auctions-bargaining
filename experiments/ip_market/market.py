import json
from typing import Dict, List, Tuple

from ..agents import call_agent_json, openai_client_from_settings
from ..settings import ModelSettings
from .models import Firm, Module


def firm_private_input(
    firm: Firm,
    modules: List[Module],
    reputation: Dict[str, float],
    last_round_deals: List[Dict],
    history: List[Dict],
    price_hints: Dict[str, Dict[str, float]],
    score: float,
    rank: int,
    total_firms: int,
) -> dict:
    owned = [
        {"tech_id": m.tech_id, "quality": m.quality}
        for m in modules
        if m.owner == firm.firm_id
    ]
    needs = [
        {
            "tech_id": m.tech_id,
            "need_level": m.needs[firm.firm_id],
            "benefit": m.benefits[firm.firm_id],
            "internal_cost": m.internal_costs[firm.firm_id],
            "integration_cost": m.integration_costs[firm.firm_id],
            "owner": m.owner,
            "suggested_bid": price_hints.get("bids", {}).get(m.tech_id),
        }
        for m in modules
    ]
    rough_signals = []
    for m in modules:
        if m.owner == firm.firm_id:
            # demand signal: count firms with need>0
            potential = [f for f, need in m.needs.items() if need > 0 and f != firm.firm_id]
            rough_signals.append(
                {
                    "tech_id": m.tech_id,
                    "potential_license_demand": "High" if len(potential) > 6 else "Medium" if potential else "Low",
                    "potential_buyers": potential[:5],
                    "suggested_ask": price_hints.get("asks", {}).get(m.tech_id),
                }
            )
        elif m.needs.get(firm.firm_id, 0) > 0:
            rough_signals.append(
                {
                    "tech_id": m.tech_id,
                    "potential_suppliers": [m.owner],
                }
            )
    return {
        "firm_id": firm.firm_id,
        "owned_technologies": owned,
        "needs": needs,
        "rough_market_signals": rough_signals,
        "transaction_cost_per_deal": 0.5,
        "reputation_score": reputation.get(firm.firm_id, 0.0),
        "last_round_deals": last_round_deals,
        "your_last_history": history[-5:],
        "your_score_last_round": score,
        "your_rank_last_round": rank,
        "total_firms": total_firms,
    }


def parse_firm_plan(raw: Dict) -> Tuple[List[Dict], List[Dict], List[Dict], str]:
    build_decisions = raw.get("build_decisions") if isinstance(raw, dict) else []
    license_sells = raw.get("license_sells") if isinstance(raw, dict) else []
    license_buys = raw.get("license_buys") if isinstance(raw, dict) else []
    commentary = raw.get("commentary", "") if isinstance(raw, dict) else ""
    return build_decisions or [], license_sells or [], license_buys or [], commentary


def double_auction(modules: List[Module], plans: Dict[str, Dict], tx_cost: float) -> Tuple[Dict, List[Tuple[str, str, str, float]]]:
    deals: List[Tuple[str, str, str, float]] = []
    use = {m.tech_id: [] for m in modules}
    # per module asks/bids
    for m in modules:
        asks = []
        bids = []
        for firm_id, plan in plans.items():
            for ask in plan["license_sells"]:
                if ask.get("tech_id") == m.tech_id:
                    asks.append((firm_id, float(ask.get("min_price", 0.0))))
            for bid in plan["license_buys"]:
                if bid.get("tech_id") == m.tech_id:
                    bids.append((firm_id, float(bid.get("max_price", 0.0))))
        if not asks or not bids:
            continue
        asks.sort(key=lambda x: x[1])
        bids.sort(key=lambda x: x[1], reverse=True)
        best_ask = asks[0]
        best_bid = bids[0]
        if best_bid[1] >= best_ask[1]:
            price = (best_bid[1] + best_ask[1]) / 2
            deals.append((m.tech_id, best_ask[0], best_bid[0], price))
            use[m.tech_id].append(best_bid[0])
    # add owners: they can use their own tech freely
    for m in modules:
        use[m.tech_id].append(m.owner)
    return use, deals


def compute_welfare(
    modules: List[Module],
    use: Dict[str, List[str]],
    plans: Dict[str, Dict],
    deals: List[Tuple[str, str, str, float]],
    tx_cost: float,
    failed: Dict[str, bool],
) -> float:
    welfare = 0.0
    # license fees net out; ignore them for welfare
    welfare -= tx_cost * len(deals)
    for tech_id, users in use.items():
        if failed.get(tech_id):
            continue
        m = next(mm for mm in modules if mm.tech_id == tech_id)
        for u in users:
            if u not in m.needs or m.needs[u] == 0:
                continue
            # if licensed (not owner), pay integration cost; if owner, integration cost 0
            integration = 0.0 if u == m.owner else m.integration_costs[u]
            welfare += m.benefits[u] - integration
    # internal builds: check plans for build_if_no_license
    for firm_id, plan in plans.items():
        build_list = plan["build_decisions"]
        for entry in build_list:
            if entry.get("decision", "") == "build_if_no_license":
                tech_id = entry.get("tech_id")
                m = next((mm for mm in modules if mm.tech_id == tech_id), None)
                if m and (firm_id not in use.get(tech_id, [])):
                    welfare += m.benefits[firm_id] - m.internal_costs[firm_id]
                    use.setdefault(tech_id, []).append(firm_id)
    return welfare


def compute_profits(
    modules: List[Module],
    use: Dict[str, List[str]],
    deals: List[Tuple[str, str, str, float]],
) -> Dict[str, float]:
    """
    Approximate per-firm profit: benefits minus integration/internal cost and payments,
    plus license revenue net of tx cost (split equally buyer/seller).
    """
    profits: Dict[str, float] = {}
    deal_price: Dict[str, float] = {t: p for t, s, b, p in deals}
    for m in modules:
        for u in use.get(m.tech_id, []):
            if u == m.owner:
                # assume internal build cost
                profits[u] = profits.get(u, 0.0) + (m.benefits[u] - m.internal_costs[u])
            else:
                price = deal_price.get(m.tech_id, 0.0)
                integration = m.integration_costs[u]
                profits[u] = profits.get(u, 0.0) + (m.benefits[u] - integration - price - 0.25)
    for t, s, b, p in deals:
        profits[s] = profits.get(s, 0.0) + p - 0.25
    return profits


def run_market(
    model_settings: ModelSettings,
    firms: List[Firm],
    modules: List[Module],
    dry_run: bool = False,
    persona_variant: str = "default",
) -> Dict:
    rounds = 2
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
            "commentary": {"type": "STRING"},
        },
    }
    reputation: Dict[str, float] = {f.firm_id: 0.0 for f in firms}
    last_round_deals: List[Dict] = []
    history_by_firm: Dict[str, List[Dict]] = {f.firm_id: [] for f in firms}
    budgets: Dict[str, float] = {f.firm_id: 50.0 for f in firms}
    round_logs: List[Dict] = []
    use_final: Dict[str, List[str]] = {}
    deals_final: List[Tuple[str, str, str, float]] = []
    profits_prev: Dict[str, float] = {f.firm_id: 0.0 for f in firms}
    for r in range(rounds):
        # simple price hints: ask ~ median benefit of others, bid ~ benefit - integration
        asks_hint: Dict[str, float] = {}
        bids_hint: Dict[str, float] = {}
        for m in modules:
            other_bens = [v for f, v in m.benefits.items() if f != m.owner]
            asks_hint[m.tech_id] = max(0.0, sum(other_bens) / len(other_bens)) if other_bens else 0.0
            for f, ben in m.benefits.items():
                if f == m.owner:
                    continue
                bids_hint[m.tech_id] = max(0.0, ben - m.integration_costs[f])
        plans: Dict[str, Dict] = {}
        # ranks based on previous profit
        sorted_firms = sorted(firms, key=lambda f: profits_prev.get(f.firm_id, 0.0), reverse=True)
        ranks = {f.firm_id: idx + 1 for idx, f in enumerate(sorted_firms)}
        for firm in firms:
            user = firm_private_input(
                firm,
                modules,
                reputation=reputation,
                last_round_deals=last_round_deals,
                history=history_by_firm[firm.firm_id],
                price_hints={"asks": asks_hint, "bids": bids_hint},
                score=profits_prev.get(firm.firm_id, 0.0),
                rank=ranks.get(firm.firm_id, len(firms)),
                total_firms=len(firms),
            )
            data, raw_text = call_agent_json(
                client=client,
                model_settings=model_settings,
                system_prompt=firm.system_prompt,
                user_prompt=json.dumps(user),
                response_schema=schema,
                dry_run=dry_run,
            )
            build_decisions, license_sells, license_buys, commentary = parse_firm_plan(data)
            # enforce at least one ask/bid where surplus exists
            if not license_sells:
                # choose one owned tech with high potential
                owned = [m for m in modules if m.owner == firm.firm_id]
                if owned:
                    target = max(owned, key=lambda mm: asks_hint.get(mm.tech_id, 0.0))
                    license_sells = [{"tech_id": target.tech_id, "min_price": max(0.0, asks_hint.get(target.tech_id, 1.0))}]
            if not license_buys:
                needs = [m for m in modules if m.needs.get(firm.firm_id, 0) > 0]
                if needs:
                    target = max(needs, key=lambda mm: bids_hint.get(mm.tech_id, 0.0))
                    license_buys = [{"tech_id": target.tech_id, "max_price": max(0.0, bids_hint.get(target.tech_id, 1.0))}]
            plans[firm.firm_id] = {
                "build_decisions": build_decisions,
                "license_sells": license_sells,
                "license_buys": license_buys,
                "commentary": commentary,
            }
        use, deals = double_auction(modules, plans, tx_cost=0.5)
        # verification: simulate failures based on quality
        failed: Dict[str, bool] = {}
        for tech_id, seller, buyer, price in deals:
            m = next(mm for mm in modules if mm.tech_id == tech_id)
            prob_fail = {"High": 0.05, "Medium": 0.2, "Low": 0.5}.get(m.quality, 0.2)
            import random

            failed[tech_id] = random.random() < prob_fail
            if failed[tech_id]:
                reputation[seller] -= 2.0
                reputation[buyer] -= 1.0
            else:
                reputation[seller] += 1.0
                reputation[buyer] += 0.5
        # small idle penalty for no asks/bids
        for firm in firms:
            if not plans[firm.firm_id]["license_buys"] and not plans[firm.firm_id]["license_sells"]:
                reputation[firm.firm_id] -= 0.5
        welfare = compute_welfare(modules, use, plans, deals, tx_cost=0.5, failed=failed)
        profits = compute_profits(modules, use, deals)
        # update budgets based on profit (clamped)
        for firm in firms:
            prof = profits.get(firm.firm_id, 0.0)
            new_budget = max(10.0, min(200.0, 25.0 + 1.0 * prof))
            budgets[firm.firm_id] = new_budget
        # record history
        for firm in firms:
            hist = {
                "round": r + 1,
                "reputation": reputation[firm.firm_id],
                "deals": [d for d in deals if d[1] == firm.firm_id or d[2] == firm.firm_id],
                "failed_deals": [d for d in deals if (d[1] == firm.firm_id or d[2] == firm.firm_id) and failed.get(d[0])],
                "profit": profits.get(firm.firm_id, 0.0),
                "budget": budgets[firm.firm_id],
            }
            history_by_firm[firm.firm_id].append(hist)
        profits_prev = profits
        round_logs.append({"round": r + 1, "plans": plans, "use": use, "deals": deals, "failed": failed, "welfare": welfare, "reputation": dict(reputation), "profits": profits})
        use_final, deals_final = use, deals
        last_round_deals = [{"tech_id": t, "seller": s, "buyer": b, "price": p, "failed": failed.get(t, False)} for t, s, b, p in deals]
    return {
        "use": use_final,
        "deals": deals_final,
        "welfare": round_logs[-1]["welfare"] if round_logs else 0.0,
        "plans": round_logs[-1]["plans"] if round_logs else {},
        "round_logs": round_logs,
    }
