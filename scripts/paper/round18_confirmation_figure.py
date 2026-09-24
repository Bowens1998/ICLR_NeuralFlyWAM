"""Plot all five scores for each primary contrast, keeping cohorts separate."""

# ruff: noqa: E402 -- standalone scientific plotting entry point.
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from round18_verify_compact import verify


def main():
    paths = {
        "original": ROOT / "reports/ROUND16_CONTROL_RESULTS.json",
        "confirmation": ROOT / "reports/ROUND18_CONTROL_RESULTS.json",
    }
    reports = {key: json.loads(path.read_text()) for key, path in paths.items()}
    verification = verify(reports["confirmation"])
    panels = {key: value["summary"]["rmse_m"] for key, value in reports.items()}
    rows = [
        ("frozen", "original"),
        ("frozen", "confirmation"),
        ("updated", "original"),
        ("updated", "confirmation"),
    ]
    colors = {"frozen": "#0072B2", "updated": "#D55E00"}
    ink, muted, grid = "#243445", "#536575", "#DCE3E8"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "flight-memory-confirmation",
            "text.color": ink,
            "axes.labelcolor": ink,
            "xtick.color": muted,
            "ytick.color": muted,
            "axes.edgecolor": grid,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.24, top=0.86, wspace=0.16)
    plotted, extent = [], [0.0]
    for axis, wind in zip(axes, ["6.1", "8.5"]):
        for index, (arm, cohort) in enumerate(rows):
            value = panels[cohort][wind]["contrasts"]["long/mixed-log/" + arm]
            points = np.asarray(value["dataset_differences"])
            assert points.shape == (5,) and np.isfinite(points).all()
            mean = value["mean"]
            lower, upper = value["descriptive_t95ci"]
            y = 3 - index
            axis.plot([lower, upper], [y, y], color=colors[arm], linewidth=1.8, zorder=2)
            axis.scatter(
                points,
                y + np.linspace(-0.12, 0.12, 5),
                s=25,
                color=colors[arm],
                alpha=0.4,
                zorder=3,
            )
            axis.scatter(
                mean,
                y,
                s=65,
                marker="o" if cohort == "original" else "D",
                facecolor="white" if cohort == "original" else colors[arm],
                edgecolor=colors[arm],
                linewidth=1.6,
                zorder=4,
            )
            extent.extend([lower, upper, *points.tolist()])
            plotted.append(
                dict(
                    wind=wind,
                    memory_design=arm,
                    cohort=cohort,
                    dataset_differences=points.tolist(),
                    mean=mean,
                    descriptive_t95ci=[lower, upper],
                )
            )
        axis.axvline(0, color=muted, linewidth=0.9, zorder=1)
        axis.axhline(1.5, color=grid, linewidth=0.8, zorder=0)
        axis.set_title(f"{wind} m/s", weight="bold", pad=10)
        axis.grid(axis="x", color=grid, linewidth=0.6, zorder=0)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.set(
            yticks=[3, 2, 1, 0],
            yticklabels=[arm.title() + " · " + cohort for arm, cohort in rows],
            ylim=(-0.45, 3.45),
        )
    padding = 0.08 * (max(extent) - min(extent))
    axes[0].set_xlim(min(extent) - padding, max(extent) + padding)
    # Matplotlib's automatic locator may return ticks outside the visible limits.
    lo, hi = axes[0].get_xlim()
    axes[0].set_xticks([x for x in axes[0].get_xticks() if lo <= x <= hi])
    fig.text(
        0.615,
        0.105,
        "Long-budget tracking ΔRMSE (m; Mixed − Re-simulated)",
        ha="center",
        fontsize=13,
    )
    fig.text(
        0.615,
        0.027,
        "Negative: Mixed improves tracking.  Each cohort uses five separate training datasets.",
        ha="center",
        fontsize=11,
        color=muted,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = list(fig.texts)
    for axis in axes:
        labels.extend([axis.title, axis.xaxis.label, axis.yaxis.label])
        labels.extend(axis.get_xticklabels() + axis.get_yticklabels())
    for label in labels:
        if not label.get_visible() or not label.get_text():
            continue
        bounds = label.get_window_extent(renderer)
        assert fig.bbox.contains(bounds.x0, bounds.y0) and fig.bbox.contains(bounds.x1, bounds.y1), label.get_text()
    output = ROOT / "paper/figures/round18_confirmation"
    for extension in ["pdf", "svg", "png"]:
        fig.savefig(output.with_suffix("." + extension), dpi=180, facecolor="white")
    plt.close(fig)
    audit = dict(
        plots=plotted,
        independent_confirmation_verification=verification,
        source_sha256={
            key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()
        },
        interpretation="Independent cohorts shown separately; no pairing between old and new datasets, no pooled inference, and no clipped dataset score or interval.",
    )
    (ROOT / "reports/ROUND18_CONFIRMATION_FIGURE.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
