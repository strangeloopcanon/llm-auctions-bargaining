from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from ..protocols import write_jsonl
from ..settings import ModelSettings
from .generator import DEFAULT_DEADLINE_ROUND, generate_firms, generate_modules_and_orders
from .market import run_brokered_market


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the brokered IP market experiment.")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=700)
    parser.add_argument("--firms", type=int, default=6)
    parser.add_argument("--modules", type=int, default=12)
    parser.add_argument("--orders", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--deadline-round", type=int, default=DEFAULT_DEADLINE_ROUND)
    parser.add_argument(
        "--provider",
        type=str,
        default="codex",
        choices=["openai", "gemini", "codex"],
    )
    parser.add_argument("--models", type=str, default="gpt-5.4,gpt-5.2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default="runs_ip_brokered_market")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--codex-home", type=str, default=None)
    parser.add_argument("--codex-reasoning-effort", type=str, default="medium")
    parser.add_argument("--codex-timeout", type=int, default=600)
    parser.add_argument(
        "--arms",
        type=str,
        default="baseline,completion_explicit",
        help="Comma-separated prompt arms.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing results file instead of overwriting it.",
    )
    return parser.parse_args()


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    models = _split_csv(args.models)
    arms = _split_csv(args.arms)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "ip_brokered_market_results.jsonl"

    rows = []
    if args.append and output_path.exists():
        rows = [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    for offset in range(args.seeds):
        seed = args.base_seed + offset
        modules, orders = generate_modules_and_orders(
            n_firms=args.firms,
            n_modules=args.modules,
            n_orders=args.orders,
            seed=seed,
            deadline_round=args.deadline_round,
        )
        for model in models:
            for arm in arms:
                firms = generate_firms(args.firms, arm=arm)
                model_settings = ModelSettings(
                    provider=args.provider,
                    model=model,
                    temperature=args.temperature,
                    codex_home=args.codex_home,
                    codex_reasoning_effort=args.codex_reasoning_effort,
                    codex_timeout_seconds=args.codex_timeout,
                )
                result = run_brokered_market(
                    model_settings=model_settings,
                    firms=firms,
                    modules=modules,
                    orders=orders,
                    dry_run=args.dry_run,
                    rounds=args.rounds,
                    seed=seed,
                )
                row = {
                    "seed": seed,
                    "arm": arm,
                    "provider": args.provider,
                    "model": model,
                    "summary": result["summary"],
                    "round_logs": result["round_logs"],
                    "orders": result["orders"],
                    "modules": result["modules"],
                }
                rows.append(row)
                write_jsonl(output_path, rows)
                print(
                    f"Completed brokered IP market seed={seed} arm={arm} model={model} "
                    f"(dry_run={args.dry_run})",
                    flush=True,
                )


if __name__ == "__main__":
    main()
