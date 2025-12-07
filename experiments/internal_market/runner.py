import argparse
from pathlib import Path

from ..settings import ModelSettings
from .simulation import SimConfig, run_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run internal market experiment.")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--quarters", type=int, default=10)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "gemini"])
    parser.add_argument("--model", type=str, default="gpt-5.1")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--output-dir", type=str, default="runs_internal_market")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_settings = ModelSettings(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
    )
    sim_cfg = SimConfig(feature_count=args.features, quarters=args.quarters)
    for i in range(args.seeds):
        seed = args.base_seed + i
        result = run_simulation(
            model_settings=model_settings,
            sim_cfg=sim_cfg,
            seed=seed,
            output_dir=Path(args.output_dir),
            dry_run=args.dry_run,
        )
        print(f"Completed internal market seed={seed}: {result}")


if __name__ == "__main__":
    main()
