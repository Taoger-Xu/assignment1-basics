"""Make standalone SVG learning curves and a throughput chart."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re


VALID_LOSS = re.compile(r"step=(\d+) valid_loss=([-+\d.eE]+)")
COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("batch_size_experiment/tinystories")
    )
    parser.add_argument("--extra-summary", type=Path, action="append", default=[])
    args = parser.parse_args()
    summary = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    results_by_batch = {int(result["batch_size"]): result for result in summary["results"]}
    for path in args.extra_summary:
        extra = json.loads(path.read_text(encoding="utf-8"))
        if extra["target_tokens"] != summary["target_tokens"]:
            raise ValueError(f"Different token budget in {path}")
        results_by_batch.update(
            {int(result["batch_size"]): result for result in extra["results"]}
        )
    results = [results_by_batch[batch] for batch in sorted(results_by_batch)]
    if not results:
        raise ValueError("No completed runs in summary.json")
    if args.extra_summary:
        (args.output_dir / "combined_summary.json").write_text(
            json.dumps({"target_tokens": summary["target_tokens"], "results": results}, indent=2),
            encoding="utf-8",
        )

    curves = {}
    for result in results:
        batch_size = int(result["batch_size"])
        log = Path(result["log"]).read_text(encoding="utf-8")
        curves[batch_size] = [
            (int(step) * batch_size * 256 / 1_000_000, float(loss))
            for step, loss in VALID_LOSS.findall(log)
            if math.isfinite(float(loss))
        ]

    width, height = 1200, 760
    x0, x1 = 100, 1110
    y0, y1 = 90, 480
    max_tokens = max(points[-1][0] for points in curves.values())
    losses = [loss for points in curves.values() for _, loss in points]
    low, high = max(0, min(losses) - 0.2), max(losses) + 0.2

    def px(tokens_m: float) -> float:
        return x0 + (x1 - x0) * tokens_m / max_tokens

    def py(loss: float) -> float:
        return y1 - (y1 - y0) * (loss - low) / (high - low)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="600" y="35" text-anchor="middle" font-size="22">TinyStories: batch size experiment</text>',
        f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#333"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#333"/>',
        '<text x="605" y="515" text-anchor="middle" font-size="15">Training tokens processed (millions)</text>',
        '<text x="30" y="285" transform="rotate(-90 30 285)" text-anchor="middle" font-size="15">Validation loss</text>',
    ]
    for index in range(6):
        value = max_tokens * index / 5
        x = px(value)
        parts.append(f'<text x="{x:.1f}" y="{y1 + 22}" text-anchor="middle" font-size="12">{value:.1f}</text>')
        value_y = low + (high - low) * index / 5
        y = py(value_y)
        parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{x0 - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12">{value_y:.1f}</text>')

    for index, result in enumerate(results):
        batch_size = int(result["batch_size"])
        color = COLORS[index % len(COLORS)]
        points = curves[batch_size]
        coords = " ".join(f"{px(tokens):.1f},{py(loss):.1f}" for tokens, loss in points)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for tokens, loss in points:
            parts.append(f'<circle cx="{px(tokens):.1f}" cy="{py(loss):.1f}" r="2.5" fill="{color}"/>')
        legend_x = 115 + index * 165
        parts.append(f'<line x1="{legend_x}" y1="548" x2="{legend_x + 22}" y2="548" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 29}" y="552" font-size="12">batch {batch_size}</text>')

    bar_y0, bar_y1 = 610, 710
    max_throughput = max(int(result["tokens_per_second"]) for result in results)
    parts.append('<text x="100" y="590" font-size="16">Throughput (tokens/s, single GPU)</text>')
    for index, result in enumerate(results):
        throughput = int(result["tokens_per_second"])
        bar_x = 110 + index * 165
        bar_height = (bar_y1 - bar_y0) * throughput / max_throughput
        parts.append(f'<rect x="{bar_x}" y="{bar_y1 - bar_height:.1f}" width="85" height="{bar_height:.1f}" fill="{COLORS[index % len(COLORS)]}"/>')
        parts.append(f'<text x="{bar_x + 42}" y="{bar_y1 - bar_height - 5:.1f}" text-anchor="middle" font-size="11">{throughput:,}</text>')
        parts.append(f'<text x="{bar_x + 42}" y="735" text-anchor="middle" font-size="12">{result["batch_size"]}</text>')
    parts.append("</svg>")
    (args.output_dir / "learning_curves.svg").write_text("\n".join(parts), encoding="utf-8")

    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("batch_size", "selected_lr", "steps", "tokens", "seconds", "tokens_per_second", "final_valid_loss"),
        )
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in writer.fieldnames})
    print(args.output_dir / "learning_curves.svg")


if __name__ == "__main__":
    main()
