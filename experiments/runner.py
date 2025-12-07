import argparse
from pathlib import Path
from typing import List

from .envs.shapley import SHAPLEY_CONDITIONS, run_shapley_condition
from .envs.vickrey import VICKREY_CONDITIONS, run_vickrey_condition
from .settings import ModelSettings, RunSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-agent mechanism experiments.")
    parser.add_argument("--mode", choices=["vickrey", "shapley"], required=True)
    parser.add_argument("--condition", default="all", help="Condition name or 'all'.")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds to run.")
    parser.add_argument("--base-seed", type=int, default=42, help="Base seed offset.")
    parser.add_argument("--output-dir", type=str, default="runs", help="Output directory.")
    parser.add_argument("--model", type=str, default="gpt-5.1")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "gemini"])
    parser.add_argument("--gemini-api-key-env", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-output-tokens", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--persona-variant", type=str, default="default", choices=["default", "sociopath", "altruist"])
    return parser.parse_args()


def collect_conditions(mode: str, name: str):
    if mode == "vickrey":
        table = VICKREY_CONDITIONS
    else:
        table = SHAPLEY_CONDITIONS
    if name == "all":
        return list(table.values())
    if name not in table:
        raise ValueError(f"Unknown condition: {name}")
    return [table[name]]


def main() -> None:
    args = parse_args()
    model_settings = ModelSettings(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        gemini_api_key_env=args.gemini_api_key_env,
    )
    run_settings = RunSettings(seeds=args.seeds, output_dir=Path(args.output_dir), dry_run=args.dry_run)
    conditions = collect_conditions(args.mode, args.condition)

    summaries: List[dict] = []
    for idx in range(run_settings.seeds):
        seed = args.base_seed + idx
        for condition in conditions:
            if args.mode == "vickrey":
                summary = run_vickrey_condition(
                    model_settings=model_settings,
                    condition=condition,
                    seed=seed,
                    output_dir=run_settings.output_dir,
                    dry_run=run_settings.dry_run,
                    persona_variant=args.persona_variant,
                )
            else:
                summary = run_shapley_condition(
                    model_settings=model_settings,
                    condition=condition,
                    seed=seed,
                    output_dir=run_settings.output_dir,
                    dry_run=run_settings.dry_run,
                    persona_variant=args.persona_variant,
                )
            summaries.append(summary)
            print(f"Completed {args.mode} condition={condition.name} seed={seed} -> {summary}")


if __name__ == "__main__":
    main()
