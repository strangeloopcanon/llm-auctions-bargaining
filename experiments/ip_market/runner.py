import argparse
from pathlib import Path

from ..settings import ModelSettings
from ..protocols import write_jsonl
from .baseline import baseline_internal_build
from .generator import generate_firms, generate_modules
from .market import run_market


def parse_args():
    p = argparse.ArgumentParser(description="Run IP licensing market experiment.")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--base-seed", type=int, default=200)
    p.add_argument("--firms", type=int, default=8)
    p.add_argument("--modules", type=int, default=20)
    p.add_argument("--provider", type=str, default="openai", choices=["openai", "gemini"])
    p.add_argument("--model", type=str, default="gpt-5.1")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--output-dir", type=str, default="runs_ip_market")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--persona-variant", type=str, default="default", choices=["default", "sociopath", "altruist"])
    p.add_argument("--adversarial-fraction", type=float, default=0.0)
    p.add_argument("--adversarial-seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    model_settings = ModelSettings(provider=args.provider, model=args.model, temperature=args.temperature)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.seeds):
        seed = args.base_seed + i
        firms = generate_firms(args.firms, adversarial_fraction=args.adversarial_fraction, seed=args.adversarial_seed + seed)
        modules, need_levels = generate_modules(args.firms, args.modules, seed=seed)
        baseline = baseline_internal_build(modules)
        market_result = run_market(
            model_settings,
            firms,
            modules,
            dry_run=args.dry_run,
            persona_variant=args.persona_variant,
        )
        write_jsonl(
            out_dir / f"ip_market_seed{seed}.jsonl",
            [
                {
                    "seed": seed,
                    "baseline": baseline,
                    "market": market_result,
                }
            ],
        )
        print(f"Completed IP market seed={seed} (dry_run={args.dry_run})")


if __name__ == "__main__":
    main()
