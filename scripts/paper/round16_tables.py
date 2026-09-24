"""Render accepted Round16 means and planned contrasts without new inference."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
PANELS = {
    p: json.loads((ROOT / f"reports/ROUND16_{p}_RESULTS.json").read_text())
    for p in ["CONTROL", "QUERY", "OFFLINE"]
}
RECIPES = ["dense", "recorded", "log", "mixed"]
LABELS = {"dense": "Dense", "recorded": "Repeated", "log": "Re-simulated", "mixed": "Mixed"}


def interval(x, digits=3):
    lo, hi = x["descriptive_t95ci"]
    return f"${x['mean']:+.{digits}f}$ & $[{lo:.{digits}f},{hi:.{digits}f}]$"


def main():
    control = PANELS["CONTROL"]["summary"]["rmse_m"]
    lines = [
        r"\begin{table}[t]\centering\small",
        r"\caption{Training recipe and budget change tracking. Failure-aware RMSE (m); learned models average five datasets, three seeds and ten episodes per wind. S/L: 5,760/21,360 steps. Physical references use the same ten episodes, without training. Figure~\ref{fig:rankings}C shows paired U--F uncertainty; Appendices~\ref{app:bridge}--\ref{app:matched_physics} retain all winds and contrasts.}\label{tab:bridge_main}",
        r"\begin{tabular}{llrrrr}\toprule",
        r"& & \multicolumn{2}{c}{6.1 m/s} & \multicolumn{2}{c}{8.5 m/s}\\",
        r"Recipe & Budget & Frozen & Updated & Frozen & Updated\\\midrule",
    ]
    for r in RECIPES:
        for b in ["short", "long"]:
            values = [control[w]["arm_means"][f"{r}_{b}/{a}"] for w in ["6.1", "8.5"] for a in ["frozen", "updated"]]
            lines.append(f"{LABELS[r]} & {'S' if b == 'short' else 'L'} & " + " & ".join(f"{v:.3f}" for v in values) + r"\\")
    physical = json.loads((ROOT / "reports/ROUND17_RESULTS.json").read_text())["summary"]
    lines.append(r"\midrule")
    for key, label in [("nominal", "Nominal-physics MPPI"), ("true_mean", "True-mean-wind MPPI")]:
        lines.append(r"\multicolumn{2}{l}{" + label + "} & " + " & ".join(
            r"\multicolumn{2}{c}{" + f"{physical[w][key]['mean']:.3f}" + "}"
            for w in ["6.1", "8.5"]) + r"\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    (PAPER / "round16_bridge_main_table.tex").write_text("\n".join(lines) + "\n")

    out = []
    panels = [
        ("CONTROL", "rmse_m", "Failure-aware tracking RMSE (m)", ["0.0", "4.9", "6.1", "8.5"], 4),
        ("QUERY", "physical_solution_cost", "Physical solution cost", ["0.0", "4.9", "6.1", "8.5"], 3),
        ("QUERY", "query_E", "Prediction error $E$ on the common candidate banks", ["0.0", "4.9", "6.1", "8.5"], 4),
        ("QUERY", "position_m", "Physical position error (m) on the common candidate banks", ["0.0", "4.9", "6.1", "8.5"], 4),
        ("OFFLINE", "original_E", "Prediction error $E$ on original static logged-action tests", ["3.7", "6.1", "8.5"], 4),
    ]
    for panel, metric, title, winds, digits in panels:
        data = PANELS[panel]["summary"] if panel == "OFFLINE" else PANELS[panel]["summary"][metric]
        out += [r"\begin{table}[!htbp]\centering\footnotesize",
                r"\caption{" + title + r". All means and paired U--F contrasts from the complete training bridge. Intervals are descriptive and nonsimultaneous across five datasets. Original checkpoints are contextual anchors, not factorial cells.}",
                r"\begin{tabular}{llrrrr}\toprule",
                r"Wind & Recipe / budget & Frozen & Updated & U--F & 95\% interval\\\midrule"]
        for wi, wind in enumerate(winds):
            cells = [f"{r}_{b}" for r in RECIPES for b in ["short", "long"]]
            if panel != "OFFLINE":
                cells.append("original")
            for cell in cells:
                label = "Original" if cell == "original" else LABELS[cell.split('_')[0]] + ' / ' + ('S' if cell.endswith('short') else 'L')
                f, u = [data[wind]['arm_means'][f'{cell}/{a}'] for a in ['frozen', 'updated']]
                out.append(f"{float(wind):g} & {label} & {f:.{digits}f} & {u:.{digits}f} & " + interval(data[wind]['contrasts'][cell+'/U-F'], digits) + r"\\")
            if wi < len(winds)-1:
                out.append(r"\midrule")
        out += [r"\bottomrule\end{tabular}\end{table}"]

    for wind in ["6.1", "8.5"]:
        out += [r"\begin{table}[!htbp]\centering\footnotesize",
                r"\caption{Complete planned training-recipe and budget contrasts in tracking RMSE (m) at " + wind + r"\,m/s. Each row uses all five paired dataset scores; negative values favor the first condition for the stated arm, or relatively favor Updated for a change in U--F.}",
                r"\begin{tabular}{lrrr}\toprule",
                r"Contrast & Mean & 95\% interval & Negative / 5\\\midrule"]
        for key, value in control[wind]['contrasts'].items():
            if key.endswith('/U-F'):
                continue
            label = key.replace('change_in_U-F', r'$\Delta$(U--F)').replace('budget_change_in_coverage_interaction', 'Long--short change in mixed--log interaction').replace('long-short', 'L--S').replace('short/', 'S: ').replace('long/', 'L: ').replace('recorded', 'Repeated').replace('log', 'Re-simulated').replace('dense', 'Dense').replace('mixed', 'Mixed').replace('frozen', 'Frozen').replace('updated', 'Updated').replace('_', ' ').replace('/', ': ')
            out.append(label + ' & ' + interval(value, 4) + f" & {value['negative_datasets']}/5" + r"\\")
        out += [r"\bottomrule\end{tabular}\end{table}"]
    (PAPER / "round16_bridge_tables.tex").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
