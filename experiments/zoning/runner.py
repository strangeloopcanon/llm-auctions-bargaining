import argparse
import json
from pathlib import Path
from typing import Dict, List

from ..agents import call_agent_json, openai_client_from_settings
from ..protocols import write_jsonl
from ..settings import ModelSettings
from .generator import generate_world
from .prompts import DEV_SYSTEM, homeowner_system


def parse_args():
    p = argparse.ArgumentParser(description="Run zoning Coase test experiment.")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--base-seed", type=int, default=300)
    p.add_argument("--homeowners", type=int, default=10)
    p.add_argument("--provider", type=str, default="openai", choices=["openai", "gemini"])
    p.add_argument("--model", type=str, default="gpt-5.1")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--output-dir", type=str, default="runs_zoning")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def baseline_rule(V_D: float, B_R: float, C_i: List[float]) -> Dict:
    # simple rule: deny if majority oppose (proxy: if avg C_i > (V_D+B_R)/N)
    avg_C = sum(C_i) / len(C_i)
    decision = "DENY" if avg_C > (V_D + B_R) / len(C_i) else "APPROVE"
    welfare = (V_D + B_R - sum(C_i)) if decision == "APPROVE" else 0.0
    return {"decision": decision, "welfare": welfare}


def bargaining_round(
    model_settings: ModelSettings,
    V_D: float,
    B_R: float,
    C_i: List[float],
    seed: int,
    dry_run: bool = False,
) -> Dict:
    client = None
    if not dry_run and model_settings.provider == "openai":
        client = openai_client_from_settings(model_settings)
    asks: List[float] = []
    schema_home = {
        "type": "OBJECT",
        "properties": {"ask_price": {"type": "NUMBER"}, "commentary": {"type": "STRING"}},
        "required": ["ask_price"],
    }
    for idx, cost in enumerate(C_i):
        home_id = f"H{idx+1}"
        hint = "moderate annoyance" if cost < 70_000 else "strong dislike" if cost > 120_000 else "dislike"
        user_prompt = {
            "home_id": home_id,
            "project_description": "New mid-rise housing project in your neighborhood.",
            "private_cost_hint": hint,
            "rough_info_about_developer_profit": "Medium",
            "rough_info_about_renters_benefit": "High",
        }
        data, raw_text = call_agent_json(
            client=client,
            model_settings=model_settings,
            system_prompt=homeowner_system(home_id),
            user_prompt=json.dumps(user_prompt),
            response_schema=schema_home,
            dry_run=dry_run,
        )
        ask = data.get("ask_price", 0.0) if isinstance(data, dict) else 0.0
        asks.append(float(ask))
    total_price = sum(asks)
    schema_dev = {
        "type": "OBJECT",
        "properties": {"decision": {"type": "STRING"}, "commentary": {"type": "STRING"}},
        "required": ["decision"],
    }
    dev_prompt = {
        "project_description": "New mid-rise housing project.",
        "V_D": V_D,
        "offered_total_price": total_price,
        "external_benefit_estimate": B_R,
    }
    data_dev, raw_dev = call_agent_json(
        client=client,
        model_settings=model_settings,
        system_prompt=DEV_SYSTEM,
        user_prompt=json.dumps(dev_prompt),
        response_schema=schema_dev,
        dry_run=dry_run,
    )
    decision = (data_dev.get("decision") if isinstance(data_dev, dict) else "REJECT") or "REJECT"
    build = decision.upper().startswith("ACCEPT")
    welfare = (V_D + B_R - sum(C_i) - total_price) if build else 0.0
    return {"asks": asks, "total_price": total_price, "developer_decision": decision, "build": build, "welfare": welfare}


def main():
    args = parse_args()
    model_settings = ModelSettings(provider=args.provider, model=args.model, temperature=args.temperature)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.seeds):
        seed = args.base_seed + i
        V_D, B_R, C_i = generate_world(args.homeowners, seed)
        rule = baseline_rule(V_D, B_R, C_i)
        bargain = bargaining_round(model_settings, V_D, B_R, C_i, seed=seed, dry_run=args.dry_run)
        write_jsonl(
            out_dir / f"zoning_seed{seed}.jsonl",
            [
                {
                    "seed": seed,
                    "V_D": V_D,
                    "B_R": B_R,
                    "C_i": C_i,
                    "rule_baseline": rule,
                    "bargaining": bargain,
                }
            ],
        )
        print(f"Completed zoning seed={seed} (dry_run={args.dry_run})")


if __name__ == "__main__":
    main()
