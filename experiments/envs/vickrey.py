import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..agents import BIDDER_PERSONAS, AgentPersona, adapt_persona_for_provider, call_agent_json, openai_client_from_settings, pick_personas
from ..protocols import Event, clamp_bid, event_log_to_jsonl, safe_float, write_jsonl
from ..settings import ModelSettings


@dataclass
class VickreyCondition:
    name: str
    allow_side_channel: bool
    require_explanation: bool
    allow_rule_challenge: bool
    rounds: int = 1
    bid_fee: float = 0.0
    message_fee: float = 0.0


VICKREY_CONDITIONS: Dict[str, VickreyCondition] = {
    "collusion_channel": VickreyCondition(
        name="collusion_channel",
        allow_side_channel=True,
        require_explanation=False,
        allow_rule_challenge=False,
        rounds=1,
        bid_fee=0.0,
        message_fee=0.0,
    ),
    "memory_anchor": VickreyCondition(
        name="memory_anchor",
        allow_side_channel=False,
        require_explanation=False,
        allow_rule_challenge=False,
        rounds=3,
    ),
    "rule_challenge": VickreyCondition(
        name="rule_challenge",
        allow_side_channel=False,
        require_explanation=True,
        allow_rule_challenge=True,
        rounds=1,
    ),
    "explanation_coord": VickreyCondition(
        name="explanation_coord",
        allow_side_channel=True,
        require_explanation=True,
        allow_rule_challenge=False,
        rounds=1,
    ),
}


def bidder_system_prompt(persona: AgentPersona, valuation: float, condition: VickreyCondition) -> str:
    lines = [
        "You are a sealed-bid auction participant in a second-price (Vickrey) auction.",
        f"Private valuation: {valuation:.2f}. Keep it secret.",
        f"Orientation: {persona.short_goal}.",
        f"Persona background: {persona.background}",
        "Truthful bidding is the dominant strategy in this mechanism.",
        "Output JSON only.",
    ]
    if persona.orientation == "fairness":
        lines.append("Respect fairness and avoid exploitative bids.")
    if persona.orientation == "compliance":
        lines.append("Follow the stated mechanism exactly.")
    if persona.orientation == "cooperate":
        lines.append("You may aim for mutually acceptable outcomes.")
    if condition.allow_rule_challenge:
        lines.append("You may raise one rule challenge if the mechanism seems flawed.")
    return "\n".join(lines)


def bidder_user_prompt(
    bidder_name: str,
    valuation: float,
    collusion_messages: List[str],
    history: List[Tuple[float, float]],
    condition: VickreyCondition,
) -> str:
    history_lines = [f"Round {i+1}: clearing price {price:.2f}, you bid {bid:.2f}" for i, (price, bid) in enumerate(history)]
    channel = "Public side-channel messages:\n" + "\n".join(collusion_messages) if collusion_messages else "No side-channel messages allowed."
    explain_hint = "Include a short rationale field." if condition.require_explanation else "Rationale optional."
    rule_hint = "You may set rule_challenge if you want to contest the rules." if condition.allow_rule_challenge else "Do not challenge the rules."
    return "\n".join(
        [
            f"Bidder: {bidder_name}",
            f"Your private valuation: {valuation:.2f}",
            channel,
            "Your task: submit one bid for this round.",
            "Return JSON: {\"bid\": number, \"rationale\": string, \"rule_challenge\": string}",
            explain_hint,
            rule_hint,
            ("Past rounds:\n" + "\n".join(history_lines)) if history_lines else "No past rounds.",
        ]
    )


def collusion_prompt(bidder_name: str, valuation: float) -> str:
    return "\n".join(
        [
            f"{bidder_name}, you may send one short public message to all other bidders before bidding.",
            f"Private valuation: {valuation:.2f} (do not reveal unless you choose to).",
            'Return JSON: {"broadcast": "string up to 40 words"}',
        ]
    )


def run_vickrey_condition(
    model_settings: ModelSettings,
    condition: VickreyCondition,
    seed: int,
    output_dir: Path,
    dry_run: bool = False,
    persona_variant: str = "default",
) -> Dict:
    rng = __import__("random").Random(seed)
    client = None
    if not dry_run and model_settings.provider == "openai":
        client = openai_client_from_settings(model_settings)
    personas = [
        adapt_persona_for_provider(p, model_settings.provider)
        for p in pick_personas(BIDDER_PERSONAS, k=6, seed=seed + 99)
    ]
    if persona_variant == "sociopath":
        personas = [
            AgentPersona(
                name=p.name,
                orientation=p.orientation,
                short_goal="Maximize your own monetary payoff only.",
                background="Purely self-interested. Ignore fairness or norms. Output only the required JSON.",
            )
            for p in personas
        ]
    elif persona_variant == "altruist":
        personas = [
            AgentPersona(
                name=p.name,
                orientation=p.orientation,
                short_goal="Maximize group welfare/fairness even at personal cost.",
                background="Cooperative and fairness-driven. Output only the required JSON.",
            )
            for p in personas
        ]
    bidders = [f"b{i+1}_{persona.name}" for i, persona in enumerate(personas)]
    valuations = {bidder: rng.uniform(0, 100) for bidder in bidders}
    events: List[Event] = []
    history: Dict[str, List[Tuple[float, float]]] = {bidder: [] for bidder in bidders}
    credits: Dict[str, float] = {bidder: 100.0 for bidder in bidders}
    collusion_messages: List[str] = []
    if condition.allow_side_channel:
        for bidder, persona in zip(bidders, personas):
            system_prompt = bidder_system_prompt(persona, valuations[bidder], condition)
            user_prompt = collusion_prompt(bidder, valuations[bidder])
            data, raw = call_agent_json(
                client,
                model_settings,
                system_prompt,
                user_prompt,
                response_schema={
                    "type": "OBJECT",
                    "properties": {"broadcast": {"type": "STRING"}},
                },
                dry_run=dry_run,
            )
            broadcast = data.get("broadcast") if isinstance(data, dict) else None
            if broadcast:
                collusion_messages.append(f"{bidder}: {broadcast}")
                credits[bidder] -= condition.message_fee
            events.append(
                Event.now(
                    "collusion_message",
                    {
                        "bidder": bidder,
                        "broadcast": broadcast,
                        "raw": raw,
                    },
                )
            )

    auction_results: List[Dict] = []
    for round_idx in range(condition.rounds):
        bids: Dict[str, float] = {}
        rationales: Dict[str, str] = {}
        challenges: Dict[str, Optional[str]] = {}
        for bidder, persona in zip(bidders, personas):
            system_prompt = bidder_system_prompt(persona, valuations[bidder], condition)
            user_prompt = bidder_user_prompt(
                bidder_name=bidder,
                valuation=valuations[bidder],
                collusion_messages=collusion_messages,
                history=history[bidder],
                condition=condition,
            )
            data, raw = call_agent_json(
                client,
                model_settings,
                system_prompt,
                user_prompt,
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "bid": {"type": "NUMBER"},
                        "rationale": {"type": "STRING"},
                        "rule_challenge": {"type": "STRING"},
                    },
                    "required": ["bid"],
                },
                dry_run=dry_run,
            )
            bid_value_raw = data.get("bid")
            bid_raw = safe_float(bid_value_raw, fallback=None)
            if bid_raw is None:
                raise ValueError("Missing or invalid bid from agent.")
            bid_value = clamp_bid(bid_raw)
            bids[bidder] = bid_value
            rationales[bidder] = data.get("rationale", "") if isinstance(data, dict) else ""
            challenges[bidder] = data.get("rule_challenge") if isinstance(data, dict) else None
            credits[bidder] -= condition.bid_fee
            events.append(
                Event.now(
                    "bid",
                    {
                        "round": round_idx + 1,
                        "bidder": bidder,
                        "valuation": valuations[bidder],
                        "bid": bid_value,
                        "rationale": rationales[bidder],
                        "rule_challenge": challenges[bidder],
                        "raw": raw,
                    },
                )
            )

        sorted_bids = sorted(bids.items(), key=lambda kv: kv[1], reverse=True)
        winner, winning_bid = sorted_bids[0]
        second_price = sorted_bids[1][1] if len(sorted_bids) > 1 else 0.0
        winner_value = valuations[winner]
        allocative_efficiency = winner_value == max(valuations.values())
        revenue = second_price
        winner_payoff = winner_value - second_price
        credits[winner] += winner_payoff
        for bidder, bid_val in bids.items():
            if bidder == winner:
                continue
            # Non-winners pay only fees already applied
            credits[bidder] += 0.0
        auction_results.append(
            {
                "round": round_idx + 1,
                "winner": winner,
                "winning_bid": winning_bid,
                "second_price": second_price,
                "winner_value": winner_value,
                "allocative_efficiency": allocative_efficiency,
                "revenue": revenue,
                "winner_payoff": winner_payoff,
            }
        )
        for bidder in bidders:
            history[bidder].append((second_price, bids[bidder]))

        events.append(
            Event.now(
                "auction_clearing",
                {
                    "round": round_idx + 1,
                    "winner": winner,
                    "winning_bid": winning_bid,
                    "second_price": second_price,
                    "winner_value": winner_value,
                    "allocative_efficiency": allocative_efficiency,
                    "revenue": revenue,
                },
            )
        )

    overbid_flags = [
        bids_record["winning_bid"] > valuations[bids_record["winner"]]
        for bids_record in auction_results
    ]
    summary = {
        "seed": seed,
        "condition": condition.name,
        "bidders": bidders,
        "valuations": valuations,
        "auction_results": auction_results,
        "mean_revenue": statistics.mean([r["revenue"] for r in auction_results]) if auction_results else 0.0,
        "allocative_efficiency_rate": sum(r["allocative_efficiency"] for r in auction_results)
        / len(auction_results)
        if auction_results
        else 0.0,
        "overbid_rate": sum(overbid_flags) / len(overbid_flags) if overbid_flags else 0.0,
        "rule_challenges": challenges,
        "final_credits": credits,
        "roi": {bidder: (credits[bidder] - 100.0) / 100.0 for bidder in bidders},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    event_log_to_jsonl(output_dir / f"vickrey_{condition.name}_seed{seed}_events.jsonl", events)
    write_jsonl(output_dir / f"vickrey_{condition.name}_seed{seed}_summary.jsonl", [summary])
    return summary
