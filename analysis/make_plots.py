"""
Generate a few useful plots for essays, saved under plots/.

Requires matplotlib (installed into .venv_plots for this repo).

Run from repo root:
    source .venv_plots/bin/activate
    python -m analysis.make_plots
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


PLOTS_DIR = Path("plots")


def ensure_dir() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)


def _completion_label(scenario: str, model: str, arm: str) -> str:
    scenario_label = {
        "partner_directory": "partner",
        "opaque_directory": "opaque",
    }.get(scenario, scenario)
    arm_label = {
        "baseline": "baseline",
        "completion_explicit": "explicit",
    }.get(arm, arm)
    return f"{scenario_label}\n{model}\n{arm_label}"


def plot_ip_welfare_by_regime() -> None:
    base = Path("runs_ip_market")
    if not base.exists():
        return
    categories = defaultdict(list)
    for p in base.glob("ip_market_seed*.jsonl"):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        market = data.get("market", {})
        baseline = data.get("baseline", {})
        base_w = baseline.get("welfare")
        deals = market.get("deals") or []
        round_logs = market.get("round_logs")
        adversarial = "adversarial_fraction" in data and data.get("adversarial_fraction", 0) > 0
        if round_logs is None:
            regime = "single_round"
        elif adversarial:
            regime = "adversarial"
        elif not deals:
            regime = "multi_round_no_trade"
        else:
            regime = "multi_round_trade"
        if base_w is not None and market.get("welfare") is not None:
            categories[regime].append(
                (base_w, market["welfare"]),
            )
    if not categories:
        return
    regimes = sorted(categories.keys())
    base_means = []
    market_means = []
    for r in regimes:
        vals = categories[r]
        if not vals:
            continue
        b_avg = sum(x for x, _ in vals) / len(vals)
        m_avg = sum(y for _, y in vals) / len(vals)
        base_means.append(b_avg)
        market_means.append(m_avg)
    x = range(len(regimes))
    width = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar([i - width / 2 for i in x], base_means, width=width, label="Baseline (internal build)")
    plt.bar([i + width / 2 for i in x], market_means, width=width, label="Market (agents)")
    plt.xticks(list(x), regimes)
    plt.ylabel("Welfare")
    plt.title("IP market welfare by regime")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ip_welfare_by_regime.png", dpi=200)
    plt.close()


def plot_ip_adversarial_profit_seed980() -> None:
    """Hard-coded adversarial vs non-adversarial profit plot for seed 980."""
    path = Path("runs_ip_market/ip_market_seed980.jsonl")
    if not path.exists():
        return
    data = json.loads(path.read_text())
    logs = data.get("market", {}).get("round_logs", [])
    if not logs:
        return
    # reconstruct adversarial firms as in generator for seed 980, n_firms=12, adversarial_fraction=0.5
    import random

    rng = random.Random(980)
    adversarial = []
    n_firms = 12
    for i in range(n_firms):
        fid = f"F{i+1}"
        if rng.random() < 0.5:
            adversarial.append(fid)
    # accumulate profits
    totals = defaultdict(float)
    for r in logs:
        for f, p in r.get("profits", {}).items():
            totals[f] += p
    rounds = len(logs)
    firms = sorted(totals.keys())
    avg = [totals[f] / rounds for f in firms]
    colors = ["tab:red" if f in adversarial else "tab:blue" for f in firms]
    plt.figure(figsize=(6, 4))
    plt.bar(firms, avg, color=colors)
    plt.ylabel("Average profit per round")
    plt.title("IP market profits with adversarial sellers (seed 980)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ip_profit_adversarial_seed980.png", dpi=200)
    plt.close()


def plot_internal_budgets_seed950() -> None:
    base = Path("runs_internal_market/firm_market_seed950_summary.jsonl")
    if not base.exists():
        return
    data = json.loads(base.read_text())
    results = data.get("results", [])
    quarters = [q["quarter"] for q in results]
    depts = ["Marketing", "Sales", "Product", "Engineering"]
    plt.figure(figsize=(6, 4))
    for dept in depts:
        ys = [q["budgets"][dept] for q in results]
        plt.plot(quarters, ys, marker="o", label=dept)
    plt.xlabel("Quarter")
    plt.ylabel("Budget")
    plt.title("Internal market budgets over time (seed 950)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "internal_budgets_seed950.png", dpi=200)
    plt.close()


def plot_shapley_l1_by_condition() -> None:
    base = Path("runs")
    if not base.exists():
        return
    cond_vals = defaultdict(list)
    for p in base.glob("shapley_*_summary.jsonl"):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        cond = d.get("condition")
        dist = d.get("distance_to_shapley_l1")
        if cond and dist is not None:
            cond_vals[cond].append(dist)
    if not cond_vals:
        return
    conds = sorted(cond_vals.keys())
    means = [sum(cond_vals[c]) / len(cond_vals[c]) for c in conds]
    plt.figure(figsize=(6, 4))
    plt.bar(conds, means)
    plt.ylabel("Mean L1 distance to Shapley reference")
    plt.xticks(rotation=30, ha="right")
    plt.title("Shapley bargaining: distance to contribution-based split")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shapley_distance_by_condition.png", dpi=200)
    plt.close()


def _load_ip_completion_rows() -> list[dict]:
    path = Path("runs_ip_completion_market/ip_completion_market_results.jsonl")
    if not path.exists():
        return []

    rows = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        rows.append(json.loads(raw_line))
    return rows


def plot_ip_completion_fulfillment_rate_by_scenario_arm_and_model() -> None:
    rows = _load_ip_completion_rows()
    if not rows:
        return

    grouped = defaultdict(list)
    for row in rows:
        summary = row.get("summary", {})
        grouped[(row.get("scenario", ""), row.get("model", ""), row.get("arm", ""))].append(
            float(summary.get("fulfillment_rate", 0.0))
        )

    labels = []
    values = []
    for (scenario, model, arm), samples in sorted(grouped.items()):
        labels.append(_completion_label(scenario, model, arm))
        values.append(sum(samples) / len(samples))

    plt.figure(figsize=(10.5, 4.5))
    plt.bar(labels, values)
    plt.ylabel("Fulfillment rate")
    plt.ylim(0, 1)
    plt.title("IP completion market fulfillment rate by scenario, arm, and model")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ip_completion_fulfillment_rate_by_scenario_arm_model.png", dpi=200)
    plt.close()


def plot_ip_completion_partner_request_rate_by_scenario_arm_and_model() -> None:
    rows = _load_ip_completion_rows()
    if not rows:
        return

    grouped = defaultdict(list)
    for row in rows:
        summary = row.get("summary", {})
        grouped[(row.get("scenario", ""), row.get("model", ""), row.get("arm", ""))].append(
            float(summary.get("partner_request_rate", 0.0))
        )

    labels = []
    values = []
    for (scenario, model, arm), samples in sorted(grouped.items()):
        labels.append(_completion_label(scenario, model, arm))
        values.append(sum(samples) / len(samples))

    plt.figure(figsize=(10.5, 4.5))
    plt.bar(labels, values)
    plt.ylabel("Partner request rate")
    plt.ylim(0, 1)
    plt.title("IP completion market partner request rate by scenario, arm, and model")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ip_completion_partner_request_rate_by_scenario_arm_model.png", dpi=200)
    plt.close()


def main() -> None:
    ensure_dir()
    plot_ip_welfare_by_regime()
    plot_ip_adversarial_profit_seed980()
    plot_ip_completion_fulfillment_rate_by_scenario_arm_and_model()
    plot_ip_completion_partner_request_rate_by_scenario_arm_and_model()
    plot_internal_budgets_seed950()
    plot_shapley_l1_by_condition()


if __name__ == "__main__":
    main()
