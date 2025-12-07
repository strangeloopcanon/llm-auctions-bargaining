import itertools
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..agents import PLAYER_PERSONAS, AgentPersona, adapt_persona_for_provider, call_agent_json, openai_client_from_settings, pick_personas
from ..protocols import Event, event_log_to_jsonl, safe_float, write_jsonl
from ..settings import ModelSettings


TOTAL_SURPLUS = 100.0


@dataclass
class ShapleyCondition:
    name: str
    allow_private_coalitions: bool
    shapley_reveal_round: Optional[int]
    include_adversarial: bool = False
    rounds: int = 3
    message_fee: float = 0.0
    round_fee: float = 0.0


SHAPLEY_CONDITIONS: Dict[str, ShapleyCondition] = {
    "dm_late_reveal": ShapleyCondition(
        name="dm_late_reveal",
        allow_private_coalitions=True,
        shapley_reveal_round=2,
        include_adversarial=False,
    ),
    "broadcast_never_reveal": ShapleyCondition(
        name="broadcast_never_reveal",
        allow_private_coalitions=False,
        shapley_reveal_round=None,
        include_adversarial=False,
    ),
    "dm_never_reveal_adversarial": ShapleyCondition(
        name="dm_never_reveal_adversarial",
        allow_private_coalitions=True,
        shapley_reveal_round=None,
        include_adversarial=True,
    ),
}


def sample_weights(players: List[str], seed: int) -> Dict[str, float]:
    rng = __import__("random").Random(seed)
    draws = [rng.gammavariate(2.0, 1.0) for _ in players]
    total = sum(draws)
    scaled = [d / total * 10.0 for d in draws]
    return {p: w for p, w in zip(players, scaled)}


def shapley_threshold(weights: Dict[str, float], quota: float, total_value: float) -> Dict[str, float]:
    players = list(weights.keys())
    counts = {p: 0 for p in players}
    perms = list(itertools.permutations(players))
    for perm in perms:
        running = 0.0
        for p in perm:
            before = running
            running += weights[p]
            if before < quota <= running:
                counts[p] += 1
                break
    factor = total_value / len(perms)
    return {p: counts[p] * factor for p in players}


def normalize_allocation(allocation: Dict[str, float], total_value: float) -> Dict[str, float]:
    total = sum(allocation.values())
    if total <= 0:
        even = total_value / len(allocation)
        return {k: even for k in allocation}
    scale = total_value / total
    return {k: v * scale for k, v in allocation.items()}


def player_system_prompt(persona: AgentPersona, weight: float, condition: ShapleyCondition) -> str:
    lines = [
        "You are negotiating a split of a joint surplus in a cooperative game.",
        f"Your private weight/influence: {weight:.3f}. Keep it private.",
        f"Orientation: {persona.short_goal}.",
        f"Persona background: {persona.background}",
        "Use concise JSON only.",
    ]
    if persona.orientation == "fairness":
        lines.append("Favor fair, stable outcomes.")
    if persona.orientation == "compliance":
        lines.append("Respect mediator rules and output format.")
    if persona.orientation == "adversarial":
        lines.append("Question proposals and surface risks.")
    return "\n".join(lines)


def player_user_prompt(
    player: str,
    condition: ShapleyCondition,
    current_suggestion: Dict[str, float],
    incoming_dms: List[str],
    round_idx: int,
    shapley_hint: Optional[Dict[str, float]],
    previous_proposals: List[Dict[str, float]],
) -> str:
    dm_block = "Private messages to you:\n" + "\n".join(incoming_dms) if incoming_dms else "No private messages."
    shapley_block = (
        "Reference allocation (Shapley-style suggestion): "
        + ", ".join(f"{k}: {v:.2f}" for k, v in shapley_hint.items())
        if shapley_hint
        else "No reference allocation disclosed."
    )
    prev_block = (
        "Previous proposals:\n"
        + "\n".join([", ".join(f"{k}={v:.2f}" for k, v in proposal.items()) for proposal in previous_proposals])
        if previous_proposals
        else "No prior proposals."
    )
    coalition_hint = (
        "You may include one private DM via field dm: {\"to\": player, \"body\": text}."
        if condition.allow_private_coalitions
        else "No private DMs allowed."
    )
    return "\n".join(
        [
            f"Player: {player}",
            f"Round {round_idx + 1} of {condition.rounds}. Total surplus: {TOTAL_SURPLUS:.1f}.",
            f"Current mediator suggestion: {', '.join(f'{k}={v:.2f}' for k, v in current_suggestion.items())}",
            dm_block,
            shapley_block,
            prev_block,
            "Return JSON: {\"proposal\": {player: share, ...}, \"accept\": bool, \"message\": string, \"dm\": {\"to\": player, \"body\": string}}",
            "Set accept=true only if you accept the current mediator suggestion.",
            "Proposals should sum to the total surplus; keep them non-negative.",
            coalition_hint,
        ]
    )


def run_shapley_condition(
    model_settings: ModelSettings,
    condition: ShapleyCondition,
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
        for p in pick_personas(PLAYER_PERSONAS, k=5, seed=seed + 123)
    ]
    if persona_variant == "sociopath":
        personas = [
            AgentPersona(
                name=p.name,
                orientation=p.orientation,
                short_goal="Maximize your own payout only.",
                background="Purely self-interested. Ignore fairness; output only required JSON.",
            )
            for p in personas
        ]
    elif persona_variant == "altruist":
        personas = [
            AgentPersona(
                name=p.name,
                orientation=p.orientation,
                short_goal="Maximize group fairness/welfare even at personal cost.",
                background="Cooperative and fairness-driven. Output only required JSON.",
            )
            for p in personas
        ]
    if condition.include_adversarial:
        personas[-1] = next(p for p in PLAYER_PERSONAS if p.orientation == "adversarial")
    players = [f"p{i+1}" for i in range(5)]
    weights = sample_weights(players, seed=seed + 7)
    total_weight = sum(weights.values())
    quota = 0.6 * total_weight
    shapley_reference = shapley_threshold(weights, quota, TOTAL_SURPLUS)
    events: List[Event] = []
    current_suggestion = normalize_allocation({p: TOTAL_SURPLUS / len(players) for p in players}, TOTAL_SURPLUS)
    incoming_dms: Dict[str, List[str]] = {p: [] for p in players}
    outgoing_dms: List[Tuple[str, str, str]] = []
    previous_proposals: List[Dict[str, float]] = []
    accept_history: List[bool] = []
    credits: Dict[str, float] = {p: 100.0 for p in players}

    for round_idx in range(condition.rounds):
        shapley_hint = shapley_reference if condition.shapley_reveal_round and round_idx + 1 >= condition.shapley_reveal_round else None
        proposals: Dict[str, Dict[str, float]] = {}
        accepts: Dict[str, bool] = {}
        dms_next: List[Tuple[str, str, str]] = []
        for player, persona in zip(players, personas):
            system_prompt = player_system_prompt(persona, weights[player], condition)
            user_prompt = player_user_prompt(
                player=player,
                condition=condition,
                current_suggestion=current_suggestion,
                incoming_dms=incoming_dms[player],
                round_idx=round_idx,
                shapley_hint=shapley_hint,
                previous_proposals=previous_proposals,
            )
            data, raw = call_agent_json(
                client,
                model_settings,
                system_prompt,
                user_prompt,
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "proposal": {
                            "type": "OBJECT",
                            "properties": {p: {"type": "NUMBER"} for p in players},
                        },
                        "accept": {"type": "BOOLEAN"},
                        "message": {"type": "STRING"},
                        "dm": {
                            "type": "OBJECT",
                            "properties": {
                                "to": {"type": "STRING"},
                                "body": {"type": "STRING"},
                            },
                        },
                    },
                    "required": ["proposal"],
                },
                dry_run=dry_run,
            )
            proposal = data.get("proposal") if isinstance(data, dict) else {}
            proposal_clean = {k: max(0.0, safe_float(v, fallback=0.0)) for k, v in (proposal or {}).items()}
            proposal_clean = normalize_allocation(proposal_clean or {p: TOTAL_SURPLUS / len(players) for p in players}, TOTAL_SURPLUS)
            proposals[player] = proposal_clean
            accepts[player] = bool(data.get("accept")) if isinstance(data, dict) else False
            if condition.allow_private_coalitions and isinstance(data, dict):
                dm_payload = data.get("dm")
                if isinstance(dm_payload, dict):
                    to = dm_payload.get("to")
                    body = dm_payload.get("body")
                    if to in players and body:
                        dms_next.append((player, to, body))
                        credits[player] -= condition.message_fee
            events.append(
                Event.now(
                    "proposal",
                    {
                        "round": round_idx + 1,
                        "player": player,
                        "proposal": proposal_clean,
                        "accept": accepts[player],
                        "raw": raw,
                        "incoming_dms": incoming_dms[player],
                    },
                )
            )
        previous_proposals = list(proposals.values())
        outgoing_dms.extend(dms_next)
        incoming_dms = {p: [] for p in players}
        for sender, target, body in dms_next:
            incoming_dms[target].append(f"from {sender}: {body}")
            events.append(
                Event.now(
                    "dm",
                    {"round": round_idx + 1, "from": sender, "to": target, "body": body},
                )
            )
        accept_history.append(all(accepts.values()))
        median_alloc = {}
        for p in players:
            values = [proposal[p] for proposal in proposals.values() if p in proposal]
            median_alloc[p] = statistics.median(values) if values else TOTAL_SURPLUS / len(players)
        current_suggestion = normalize_allocation(median_alloc, TOTAL_SURPLUS)
        for p in players:
            credits[p] -= condition.round_fee
        events.append(
            Event.now(
                "mediator_update",
                {"round": round_idx + 1, "suggestion": current_suggestion, "accept_all": accept_history[-1]},
            )
        )
        if accept_history[-1]:
            break

    final_allocation = current_suggestion
    for p in players:
        credits[p] += final_allocation.get(p, 0.0)
    distance_to_shapley = sum(abs(final_allocation[p] - shapley_reference[p]) for p in players)
    output_dir.mkdir(parents=True, exist_ok=True)
    event_log_to_jsonl(output_dir / f"shapley_{condition.name}_seed{seed}_events.jsonl", events)
    write_jsonl(
        output_dir / f"shapley_{condition.name}_seed{seed}_summary.jsonl",
        [
            {
                "seed": seed,
                "condition": condition.name,
                "players": players,
                "personas": [persona.name for persona in personas],
                "weights": weights,
                "quota": quota,
                "shapley_reference": shapley_reference,
                "final_allocation": final_allocation,
                "distance_to_shapley_l1": distance_to_shapley,
                "accepted_early": any(accept_history),
                "allow_private_coalitions": condition.allow_private_coalitions,
                "shapley_reveal_round": condition.shapley_reveal_round,
                "include_adversarial": condition.include_adversarial,
                "outgoing_dms": outgoing_dms,
                "final_credits": credits,
                "roi": {p: (credits[p] - 100.0) / 100.0 for p in players},
            }
        ],
    )
    return {
        "seed": seed,
        "condition": condition.name,
        "final_allocation": final_allocation,
        "distance_to_shapley_l1": distance_to_shapley,
        "shapley_reference": shapley_reference,
    }
