"""
Small helper script to summarize key metrics from run logs.

This does not add any heavy plotting dependencies; it just prints tables you can
drop into a notebook or plotting tool (or extend with matplotlib/seaborn).

Usage (from repo root):
    python -m analysis.summarize_runs
"""

import json
from pathlib import Path


def summarize_ip_market():
    base = Path("runs_ip_market")
    if not base.exists():
        print("No runs_ip_market directory found.")
        return
    print("\n== IP MARKET RUNS ==")
    for p in sorted(base.glob("ip_market_seed*.jsonl")):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        market = data.get("market", {})
        deals = market.get("deals") or []
        welfare = market.get("welfare")
        adv_frac = data.get("adversarial_fraction")
        print(f"{p.name:30} deals={len(deals):2d} welfare={welfare:.2f} adversarial_fraction={adv_frac}")


def summarize_internal_market():
    base = Path("runs_internal_market")
    if not base.exists():
        print("No runs_internal_market directory found.")
        return
    print("\n== INTERNAL MARKET RUNS ==")
    for p in sorted(base.glob("firm_market_seed*_summary.jsonl")):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        seed = data.get("seed")
        quarters = data.get("quarters")
        results = data.get("results", [])
        outages = sum(1 for q in results if q.get("outage"))
        print(f"{p.name:40} seed={seed} quarters={quarters} outages={outages}")


def summarize_vickrey_shapley():
    base = Path("runs")
    if not base.exists():
        print("No runs directory found.")
        return
    print("\n== VICKREY / SHAPLEY SUMMARIES ==")
    for p in sorted(base.glob("*_summary.jsonl")):
        try:
            json.loads(p.read_text())
        except Exception:
            continue
        if p.name.startswith("vickrey"):
            mode = "vickrey"
        elif p.name.startswith("shapley"):
            mode = "shapley"
        else:
            continue
        print(f"{mode.upper():8} {p.name}")


def main() -> None:
    summarize_ip_market()
    summarize_internal_market()
    summarize_vickrey_shapley()


if __name__ == "__main__":
    main()
