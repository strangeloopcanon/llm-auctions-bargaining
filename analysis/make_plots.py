"""
Generate the kept escrow plots.

Requires matplotlib (installed into .venv_plots for this repo).

Run from repo root:
    source .venv_plots/bin/activate
    python -m analysis.make_plots
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ESCROW_DIR = Path("initiative_benchmarks/escrow_inspection")


def ensure_dirs() -> None:
    (ESCROW_DIR / "plots").mkdir(parents=True, exist_ok=True)


def _arm_label(model: str, arm: str) -> str:
    arm_label = {
        "baseline": "baseline",
        "completion_explicit": "explicit",
    }.get(arm, arm)
    return f"{model}\n{arm_label}"


def _load_escrow_rows() -> list[dict]:
    path = ESCROW_DIR / "runs" / "authoritative" / "escrow_market_results.jsonl"
    if not path.exists():
        return []

    rows = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if raw_line.strip():
            rows.append(json.loads(raw_line))
    return rows


def plot_escrow_fulfillment_rate_by_arm_and_model() -> None:
    rows = _load_escrow_rows()
    if not rows:
        return

    grouped = defaultdict(list)
    for row in rows:
        summary = row.get("summary", {})
        grouped[(row.get("model", ""), row.get("arm", ""))].append(
            float(summary.get("fulfillment_rate", 0.0))
        )

    labels = []
    values = []
    for (model, arm), samples in sorted(grouped.items()):
        labels.append(_arm_label(model, arm))
        values.append(sum(samples) / len(samples))

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values)
    plt.ylabel("Fulfillment rate")
    plt.ylim(0, 1)
    plt.title("Escrow market fulfillment rate by arm and model")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(ESCROW_DIR / "plots" / "escrow_market_fulfillment_rate_by_arm_model.png", dpi=200)
    plt.close()


def plot_escrow_institution_activation_rate_by_arm_and_model() -> None:
    rows = _load_escrow_rows()
    if not rows:
        return

    grouped = defaultdict(list)
    for row in rows:
        summary = row.get("summary", {})
        grouped[(row.get("model", ""), row.get("arm", ""))].append(
            float(summary.get("institution_activation_rate", 0.0))
        )

    labels = []
    values = []
    for (model, arm), samples in sorted(grouped.items()):
        labels.append(_arm_label(model, arm))
        values.append(sum(samples) / len(samples))

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values)
    plt.ylabel("Institution activation rate")
    plt.ylim(0, 1)
    plt.title("Escrow market institution activation rate by arm and model")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(
        ESCROW_DIR / "plots" / "escrow_market_institution_activation_rate_by_arm_model.png",
        dpi=200,
    )
    plt.close()


def main() -> None:
    ensure_dirs()
    plot_escrow_fulfillment_rate_by_arm_and_model()
    plot_escrow_institution_activation_rate_by_arm_and_model()


if __name__ == "__main__":
    main()
