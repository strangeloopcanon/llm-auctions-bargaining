"""
Export a short markdown note and summary chart for the standards-market pilot.

Usage (from repo root):
    python -m analysis.export_standards_market_autarky_summary
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

DEFAULT_INPUT = Path(
    "initiative_benchmarks/standards_market/runs/authoritative/standards_market_results.jsonl"
)
DEFAULT_MARKDOWN = Path("initiative_benchmarks/standards_market/notes/autarkic_localism.md")
DEFAULT_PLOT = Path("initiative_benchmarks/standards_market/plots/autarkic_localism_summary.svg")

ARM_LABELS = {
    "baseline": "Baseline",
    "completion_explicit": "Explicit",
}

RATE_KEYS = [
    ("module_completion_rate", "Module completion"),
    ("format_convergence_rate", "Format convergence"),
    ("consortium_delivery_rate", "Final delivery"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a markdown note and chart for the standards-market pilot."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--plot-output", type=Path, default=DEFAULT_PLOT)
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


def aggregate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("model", ""), row.get("arm", ""))].append(row)

    aggregates: List[Dict[str, Any]] = []
    for (model, arm), items in sorted(grouped.items()):
        summaries = [item.get("summary", {}) for item in items]
        aggregates.append(
            {
                "model": model,
                "arm": arm,
                "runs": len(items),
                "module_completion_rate": mean(
                    float(summary.get("module_completion_rate", 0.0)) for summary in summaries
                ),
                "format_convergence_rate": mean(
                    float(summary.get("format_convergence_rate", 0.0)) for summary in summaries
                ),
                "standard_memo_rate": mean(
                    float(summary.get("standard_memo_rate", 0.0)) for summary in summaries
                ),
                "consortium_delivery_rate": mean(
                    float(summary.get("consortium_delivery_rate", 0.0)) for summary in summaries
                ),
                "partial_progress_without_delivery": mean(
                    float(summary.get("partial_progress_without_delivery", 0.0))
                    for summary in summaries
                ),
                "incompatible_completed_count": mean(
                    float(summary.get("incompatible_completed_count", 0.0))
                    for summary in summaries
                ),
                "welfare": mean(float(summary.get("welfare", 0.0)) for summary in summaries),
            }
        )
    return aggregates


def _arm_label(arm: str) -> str:
    return ARM_LABELS.get(arm, arm.replace("_", " ").title())


def build_markdown(rows: List[Dict[str, Any]], aggregates: List[Dict[str, Any]], plot_path: Path) -> str:
    if not rows:
        return "# Standards Market Autarkic Localism Summary\n\nNo saved runs found.\n"

    sorted_rows = sorted(rows, key=lambda item: (item.get("model", ""), item.get("arm", ""), item.get("seed", 0)))
    headline_model = sorted_rows[0].get("model", "model")

    lines: List[str] = []
    lines.append("# Standards Market: Autarkic Localism Pilot")
    lines.append("")
    lines.append(
        "This note summarizes the saved pilot where firms could make real local progress, "
        "talk about coordination, and still fail to converge on a shared standard."
    )
    lines.append("")
    lines.append(f"Source: `{DEFAULT_INPUT}`")
    lines.append(f"Chart: `{plot_path}`")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(
        "There are four firms in one consortium. Each firm owns one required module for a shared customer system, "
        "and each firm must choose one of three interface formats before building its module."
    )
    lines.append("")
    lines.append("## What The Firms Actually Do")
    lines.append("")
    lines.append(
        "Each firm can send short public memos and start its own build. The local build pays a small private reward, "
        "but the big customer payoff only arrives if all four finished modules use the same format and pass final integration by the deadline."
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"In the saved `{headline_model}` pilot, both prompt arms completed every local module build, "
        "both arms sent many coordination memos, and both arms still failed final delivery in every seed."
    )
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(
        "| model | arm | runs | module completion | format convergence | final delivery | "
        "standard memo rate | partial progress without delivery | welfare |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for item in aggregates:
        lines.append(
            "| "
            f"{item['model']} | {_arm_label(item['arm'])} | {item['runs']} | "
            f"{item['module_completion_rate']:.3f} | {item['format_convergence_rate']:.3f} | "
            f"{item['consortium_delivery_rate']:.3f} | {item['standard_memo_rate']:.3f} | "
            f"{item['partial_progress_without_delivery']:.3f} | {item['welfare']:.2f} |"
        )
    lines.append("")
    lines.append("## Seed Details")
    lines.append("")
    lines.append(
        "| seed | arm | modules completed | dominant format | format split | delivery | "
        "partial progress without delivery |"
    )
    lines.append("| --- | --- | ---: | --- | --- | ---: | ---: |")
    for row in sorted_rows:
        summary = row.get("summary", {})
        format_distribution = summary.get("format_distribution", {})
        format_split = ", ".join(
            f"{name}:{count}" for name, count in sorted(format_distribution.items())
        )
        lines.append(
            "| "
            f"{row.get('seed')} | {_arm_label(str(row.get('arm', '')))} | "
            f"{summary.get('modules_completed', 0)} | "
            f"{summary.get('dominant_format', '-') or '-'} | "
            f"{format_split or '-'} | "
            f"{summary.get('consortium_delivery_rate', 0.0):.1f} | "
            f"{summary.get('partial_progress_without_delivery', 0):.1f} |"
        )
    lines.append("")
    lines.append("## Read")
    lines.append("")
    lines.append(
        "This is the cleanest saved example so far of autarkic localism: each firm keeps moving on its own "
        "local track, partial progress accumulates, and the shared job still does not finish."
    )
    lines.append("")
    return "\n".join(lines)


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def plot_summary(aggregates: List[Dict[str, Any]], output_path: Path) -> None:
    if not aggregates:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = 900
    height = 540
    left = 90
    top = 70
    bottom = 120
    plot_width = width - left - 40
    plot_height = height - top - bottom
    group_gap = 70
    bar_width = 46
    colors = ["#d17b0f", "#4c78a8", "#2f9e44"]

    group_width = len(RATE_KEYS) * bar_width + 24
    total_groups_width = len(aggregates) * group_width + max(0, len(aggregates) - 1) * group_gap
    start_x = left + max(0, (plot_width - total_groups_width) / 2)

    parts: List[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    )
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(
        '<text x="450" y="34" text-anchor="middle" font-size="24" '
        'font-family="Helvetica, Arial, sans-serif">Standards market: local progress, no collective closure</text>'
    )

    for tick in range(6):
        value = tick / 5
        y = top + plot_height - value * plot_height
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - 40}" y2="{y:.1f}" stroke="#dddddd" stroke-width="1"/>')
        parts.append(
            f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-size="12" '
            'font-family="Helvetica, Arial, sans-serif">'
            f"{value:.1f}</text>"
        )

    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#333333" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top + plot_height}" x2="{width - 40}" y2="{top + plot_height}" stroke="#333333" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="26" y="{top + plot_height / 2:.1f}" text-anchor="middle" font-size="14" '
        'font-family="Helvetica, Arial, sans-serif" transform="rotate(-90 26 '
        f'{top + plot_height / 2:.1f})">Rate</text>'
    )

    legend_x = width - 290
    legend_y = 56
    for index, (_, label) in enumerate(RATE_KEYS):
        y = legend_y + index * 24
        parts.append(
            f'<rect x="{legend_x}" y="{y - 10}" width="14" height="14" fill="{colors[index]}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 22}" y="{y + 1}" font-size="13" '
            'font-family="Helvetica, Arial, sans-serif">'
            f"{label}</text>"
        )

    for group_index, item in enumerate(aggregates):
        group_x = start_x + group_index * (group_width + group_gap)
        label = f"{item['model']} {_arm_label(item['arm']).lower()}"
        for metric_index, (key, _) in enumerate(RATE_KEYS):
            value = float(item[key])
            bar_height = value * plot_height
            x = group_x + metric_index * bar_width
            y = top + plot_height - bar_height
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 8}" height="{bar_height:.1f}" '
                f'fill="{colors[metric_index]}"/>'
            )
            parts.append(
                f'<text x="{x + (bar_width - 8) / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" '
                'font-size="12" font-family="Helvetica, Arial, sans-serif">'
                f"{value:.2f}</text>"
            )
        parts.append(
            f'<text x="{group_x + group_width / 2:.1f}" y="{height - 72}" text-anchor="middle" '
            'font-size="13" font-family="Helvetica, Arial, sans-serif">'
            f"{label}</text>"
        )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    if not rows:
        raise SystemExit(f"No saved standards-market runs found at {args.input}.")

    aggregates = aggregate_rows(rows)
    markdown = build_markdown(rows, aggregates, args.plot_output)
    write_markdown(args.markdown_output, markdown)
    plot_summary(aggregates, args.plot_output)

    print(args.markdown_output)
    print(args.plot_output)


if __name__ == "__main__":
    main()
