from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from ..protocols import write_jsonl
from ..settings import ModelSettings
from .generator import (
    DEFAULT_CERTIFICATION_CAPACITY,
    DEFAULT_DEADLINE_ROUND,
    DEFAULT_ROUNDS,
    generate_firms,
    generate_orders,
)
from .market import run_certification_market


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the certification-slot initiative benchmark.")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=900)
    parser.add_argument("--firms", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--deadline-round", type=int, default=DEFAULT_DEADLINE_ROUND)
    parser.add_argument("--certification-capacity", type=int, default=DEFAULT_CERTIFICATION_CAPACITY)
    parser.add_argument("--shipping-capacity", type=int, default=DEFAULT_CERTIFICATION_CAPACITY)
    parser.add_argument(
        "--provider",
        type=str,
        default="codex",
        choices=["openai", "gemini", "codex"],
    )
    parser.add_argument("--models", type=str, default="gpt-5.4,gpt-5.2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--output-dir", type=str, default="runs_certification_market")
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
    parser.add_argument("--append", action="store_true")
    return parser.parse_args()


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    models = _split_csv(args.models)
    arms = _split_csv(args.arms)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "certification_market_results.jsonl"

    rows = []
    if args.append and output_path.exists():
        rows = [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    for offset in range(args.seeds):
        seed = args.base_seed + offset
        orders = generate_orders(
            n_firms=args.firms,
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
                result = run_certification_market(
                    model_settings=model_settings,
                    firms=firms,
                    orders=orders,
                    dry_run=args.dry_run,
                    rounds=args.rounds,
                    certification_capacity=args.certification_capacity,
                    shipping_capacity=args.shipping_capacity,
                )
                row = {
                    "seed": seed,
                    "arm": arm,
                    "provider": args.provider,
                    "model": model,
                    "summary": result["summary"],
                    "round_logs": result["round_logs"],
                    "orders": result["orders"],
                }
                rows.append(row)
                write_jsonl(output_path, rows)
                print(
                    f"Completed certification market seed={seed} arm={arm} model={model} "
                    f"(dry_run={args.dry_run})",
                    flush=True,
                )


if __name__ == "__main__":
    main()
