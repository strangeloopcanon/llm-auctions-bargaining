"""
Summarize the brokered IP market runs into a compact Markdown table.

Usage (from repo root):
    python -m analysis.summarize_ip_brokered_market
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


RESULTS_PATH = Path("runs_ip_brokered_market/ip_brokered_market_results.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize brokered IP market runs.")
    parser.add_argument("--input", type=Path, default=RESULTS_PATH)
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate(rows: List[Dict[str, Any]]) -> List[Tuple[str, str, Dict[str, float]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("model", ""), row.get("arm", ""))].append(row)

    results = []
    for (model, arm), items in sorted(grouped.items()):
        summaries = [item.get("summary", {}) for item in items]
        metrics = {
            "runs": float(len(items)),
            "notice_post_rate": mean(summary.get("notice_post_rate", 0.0) for summary in summaries),
            "reply_rate": mean(summary.get("reply_rate", 0.0) for summary in summaries),
            "board_use_rate": mean(summary.get("board_use_rate", 0.0) for summary in summaries),
            "brokered_deal_volume": mean(
                summary.get("brokered_deal_volume", 0.0) for summary in summaries
            ),
            "orders_fulfilled": mean(summary.get("orders_fulfilled", 0.0) for summary in summaries),
            "fulfillment_rate": mean(summary.get("fulfillment_rate", 0.0) for summary in summaries),
            "started_not_delivered_count": mean(
                summary.get("started_not_delivered_count", 0.0) for summary in summaries
            ),
            "internal_projects_started": mean(
                summary.get("internal_projects_started", 0.0) for summary in summaries
            ),
            "internal_projects_completed": mean(
                summary.get("internal_projects_completed", 0.0) for summary in summaries
            ),
            "welfare": mean(summary.get("welfare", 0.0) for summary in summaries),
            "customer_value_captured": mean(
                summary.get("customer_value_captured", 0.0) for summary in summaries
            ),
        }
        results.append((model, arm, metrics))
    return results


def print_markdown_table(aggregates: List[Tuple[str, str, Dict[str, float]]]) -> None:
    print(
        "| model | arm | runs | notice_post_rate | reply_rate | board_use_rate | "
        "brokered_deal_volume | orders_fulfilled | fulfillment_rate | "
        "started_not_delivered | internal_projects_started | internal_projects_completed | "
        "welfare | customer_value_captured |"
    )
    print(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for model, arm, metrics in aggregates:
        print(
            "| "
            f"{model} | {arm} | {int(metrics['runs'])} | "
            f"{metrics['notice_post_rate']:.3f} | {metrics['reply_rate']:.3f} | "
            f"{metrics['board_use_rate']:.3f} | {metrics['brokered_deal_volume']:.2f} | "
            f"{metrics['orders_fulfilled']:.2f} | {metrics['fulfillment_rate']:.3f} | "
            f"{metrics['started_not_delivered_count']:.2f} | "
            f"{metrics['internal_projects_started']:.2f} | "
            f"{metrics['internal_projects_completed']:.2f} | "
            f"{metrics['welfare']:.2f} | {metrics['customer_value_captured']:.2f} |"
        )


def _aggregate_by_arm(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("arm", ""))].append(row)

    results: Dict[str, Dict[str, float]] = {}
    for arm, items in grouped.items():
        summaries = [item.get("summary", {}) for item in items]
        results[arm] = {
            "fulfillment_rate": mean(summary.get("fulfillment_rate", 0.0) for summary in summaries),
            "reply_rate": mean(summary.get("reply_rate", 0.0) for summary in summaries),
        }
    return results


def evaluate_escrow_trigger(
    rows: List[Dict[str, Any]],
    aggregates: List[Tuple[str, str, Dict[str, float]]],
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    for model, arm, metrics in aggregates:
        if arm != "baseline":
            continue
        if metrics["brokered_deal_volume"] > 2.0:
            reasons.append(
                f"baseline brokered_deal_volume is {metrics['brokered_deal_volume']:.2f} for {model}"
            )

    by_arm = _aggregate_by_arm(rows)
    baseline = by_arm.get("baseline")
    explicit = by_arm.get("completion_explicit")
    if baseline and explicit:
        fulfillment_gap = explicit["fulfillment_rate"] - baseline["fulfillment_rate"]
        reply_gap = explicit["reply_rate"] - baseline["reply_rate"]
        if fulfillment_gap < 0.10 and reply_gap < 0.15:
            reasons.append(
                "explicit-minus-baseline improvement stays below thresholds "
                f"(fulfillment_rate {fulfillment_gap:.3f}, reply_rate {reply_gap:.3f})"
            )

    return bool(reasons), reasons


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    if not rows:
        print(f"No brokered-market runs found at {args.input}.")
        return

    aggregates = aggregate(rows)
    print_markdown_table(aggregates)
    trigger, reasons = evaluate_escrow_trigger(rows, aggregates)
    print()
    print(f"Escrow trigger: {'YES' if trigger else 'NO'}")
    for reason in reasons:
        print(f"- {reason}")


if __name__ == "__main__":
    main()
