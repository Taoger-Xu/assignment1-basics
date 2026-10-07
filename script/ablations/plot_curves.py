"""Plot validation learning curves from the baseline and ablation console logs."""

from __future__ import annotations

import argparse
import json
import math
import re
from html import escape
from pathlib import Path


VALIDATION = re.compile(r"step=(\d+) valid_loss=([-+\d.eE]+)")
PANELS = (
    ("RMSNorm / learning rate", ("baseline", "no_rmsnorm_base_lr", "no_rmsnorm_mid_lr", "no_rmsnorm_low_lr")),
    ("Norm placement and positions", ("baseline", "post_norm", "nope")),
    ("Feed-forward activation", ("baseline", "silu")),
)
COLORS = {
    "baseline": "#1f77b4",
    "no_rmsnorm_base_lr": "#d62728",
    "no_rmsnorm_mid_lr": "#ff7f0e",
    "no_rmsnorm_low_lr": "#9467bd",
    "post_norm": "#2ca02c",
    "nope": "#e377c2",
    "silu": "#8c564b",
}


def read_validation(path: Path) -> list[tuple[int, float]]:
    text = path.read_text(encoding="utf-8")
    return [
        (int(step), float(loss))
        for step, loss in VALIDATION.findall(text)
        if math.isfinite(float(loss))
    ]


def render_panel(
    title: str,
    names: tuple[str, ...],
    curves: dict[str, list[tuple[int, float]]],
    y_top: int,
    max_step: int,
) -> str:
    left, right = 95, 1040
    top, bottom = y_top + 44, y_top + 190
    values = [loss for name in names for step, loss in curves[name] if step <= max_step]
    if not values:
        return ""
    low, high = min(values), max(values)
    padding = max(0.08, (high - low) * 0.12)
    low = max(0, low - padding)
    high += padding
    clipped = high > 8.0
    high = min(high, 8.0)
    low = min(low, high - 0.08)

    def xcoord(step: int) -> float:
        return left + (right - left) * step / max_step

    def ycoord(loss: float) -> float:
        return bottom - (bottom - top) * (min(loss, high) - low) / (high - low)

    parts = [
        f'<text x="95" y="{y_top + 20}" font-size="18" font-weight="bold">{escape(title)}</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#333"/>',
        f'<text x="20" y="{top + 5}" font-size="12">{high:.2f}</text>',
        f'<text x="20" y="{bottom}" font-size="12">{low:.2f}</text>',
    ]
    if clipped:
        parts.append(
            f'<text x="{right}" y="{top + 12}" text-anchor="end" font-size="11" fill="#d62728">loss above 8 clipped (divergence)</text>'
        )
    for step in range(0, max_step + 1, 1000):
        x = xcoord(step)
        parts.append(f'<text x="{x:.1f}" y="{bottom + 19}" text-anchor="middle" font-size="11">{step}</text>')
    for index, name in enumerate(names):
        points = [(step, loss) for step, loss in curves[name] if step <= max_step]
        color = COLORS[name]
        if points:
            polyline = " ".join(f"{xcoord(step):.1f},{ycoord(loss):.1f}" for step, loss in points)
            parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for step, loss in points:
                parts.append(f'<circle cx="{xcoord(step):.1f}" cy="{ycoord(loss):.1f}" r="3" fill="{color}"/>')
        legend_x = left + index * 235
        parts.append(f'<line x1="{legend_x}" y1="{bottom + 44}" x2="{legend_x + 20}" y2="{bottom + 44}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 27}" y="{bottom + 48}" font-size="12">{escape(name)}</text>')
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-log", type=Path, default=Path("swanlog/run-20261007_002831-0rtuabuq/console/2026-10-07.log"))
    parser.add_argument("--output-dir", type=Path, default=Path("ablations/tinystories"))
    parser.add_argument("--max-step", type=int, default=3000)
    args = parser.parse_args()

    curves = {"baseline": read_validation(args.baseline_log)}
    for name in COLORS:
        if name != "baseline":
            curves[name] = read_validation(args.output_dir / f"{name}.log")

    height = 770
    body = "\n".join(
        render_panel(title, names, curves, 35 + index * 240, args.max_step)
        for index, (title, names) in enumerate(PANELS)
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        '<text x="550" y="26" text-anchor="middle" font-size="20">TinyStories ablations: validation cross-entropy vs. step</text>\n'
        f'{body}\n</svg>\n'
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation_curves.svg").write_text(svg, encoding="utf-8")
    (args.output_dir / "validation_curves.json").write_text(
        json.dumps(curves, indent=2), encoding="utf-8"
    )
    print(args.output_dir / "validation_curves.svg")


if __name__ == "__main__":
    main()
