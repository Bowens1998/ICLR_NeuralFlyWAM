"""Render complete matched-reference scores and paired mainline comparisons."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    data = json.loads((ROOT / "reports/ROUND17_RESULTS.json").read_text())
    lines = [
        r"\begin{table}[!htbp]\centering\small",
        r"\caption{Matched physical references and long-budget Mixed models. Failure-aware tracking RMSE (m), on identical episodes. Physical-reference means average ten episodes; learned means also average all five datasets and three model seeds.}",
        r"\begin{tabular}{lrrrr}\toprule",
        r"Wind (m/s) & \shortstack{Nominal-physics\\MPPI} & \shortstack{True-mean-wind\\MPPI} & Mixed Frozen & Mixed Updated\\\midrule",
    ]
    learned = json.loads((ROOT / "reports/ROUND16_CONTROL_RESULTS.json").read_text())["summary"][
        "rmse_m"
    ]
    for w in ["0.0", "4.9", "6.1", "8.5"]:
        values = [data["summary"][w][k]["mean"] for k in ["nominal", "true_mean"]]
        values += [learned[w]["arm_means"]["mixed_long/" + a] for a in ["frozen", "updated"]]
        lines.append(w + " & " + " & ".join(f"{v:.4f}" for v in values) + r"\\")
    lines += [
        r"\bottomrule\end{tabular}\end{table}",
        r"\begin{table}[!htbp]\centering\small",
        r"\caption{Long-budget Mixed minus each matched reference. Negative tracking RMSE differences (m) favor the learned model. All intervals use five training datasets, conditional on the shared episode roster. The complete 144-contrast roster is retained in the compact supplement.}",
        r"\begin{tabular}{llrrr}\toprule",
        r"Wind & Memory design & Reference & Difference & 95\% interval\\\midrule",
    ]
    for w in ["0.0", "4.9", "6.1", "8.5"]:
        for arm in ["frozen", "updated"]:
            for k, label in [
                ("nominal", "Nominal-physics MPPI"),
                ("true_mean", "True-mean-wind MPPI"),
            ]:
                x = data["summary"][w][k]["learned_minus_reference"]["mixed_long/" + arm]
                lo, hi = x["descriptive_t95ci"]
                lines.append(
                    f"{w} & {arm.title()} & {label} & ${x['mean']:+.4f}$ & $[{lo:.4f},{hi:.4f}]$"
                    + r"\\"
                )
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    (ROOT / "paper/round17_tables.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
