"""Primary plots. Matplotlib defaults only -- no styling dependencies."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_error_vs_horizon(results: dict[str, dict], out_path: Path, component: str = "aggregate"):
    """results: {label: summary dict}. One line per model."""
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for label, s in results.items():
        ax.plot(s["horizon_seconds"], s["per_horizon"][component], label=label, lw=1.8)
    ax.set_xlabel("prediction horizon [s]")
    ax.set_ylabel(f"normalised {component} error")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_time_aligned(
    per_window: dict[str, dict],
    out_path: Path,
    flight: str,
    dt: float,
    horizon_index: int = -1,
    component: str = "aggregate",
    smooth: int = 25,
):
    """Error at a fixed horizon as a function of time along one flight.

    This is the changing-wind figure: the latent is re-estimated at every step,
    so the curve shows how quickly each model recovers after the wind changes.
    """
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for label, pw in per_window.items():
        names = list(pw["flight_names"])
        if flight not in names:
            continue
        fid = names.index(flight)
        m = pw["flight_ids"] == fid
        t = pw["centre_index"][m] * dt
        y = pw[f"err_{component}"][m][:, horizon_index]
        order = np.argsort(t)
        t, y = t[order], y[order]
        if smooth > 1 and y.size > smooth:
            k = np.ones(smooth) / smooth
            y = np.convolve(y, k, mode="same")
        ax.plot(t, y, label=label, lw=1.3)
    ax.set_xlabel("time along flight [s]")
    ax.set_ylabel(f"normalised {component} error @ terminal horizon")
    ax.set_title(flight, fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_split_comparison(table: list[dict], out_path: Path, splits: list[str]):
    """Grouped bars: mean aggregate error per model per split, with seed spread."""
    models = sorted({r["model"] for r in table})
    fig, ax = plt.subplots(figsize=(1.6 * len(splits) + 2.5, 3.4))
    width = 0.8 / max(len(models), 1)
    x = np.arange(len(splits))
    for i, model in enumerate(models):
        means, errs = [], []
        for sp in splits:
            vals = [r["aggregate"] for r in table if r["model"] == model and r["split"] == sp]
            means.append(np.mean(vals) if vals else np.nan)
            errs.append(np.std(vals) if len(vals) > 1 else 0.0)
        ax.bar(x + i * width - 0.4 + width / 2, means, width, yerr=errs, capsize=3, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontsize=8)
    ax.set_ylabel("normalised aggregate rollout error")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
