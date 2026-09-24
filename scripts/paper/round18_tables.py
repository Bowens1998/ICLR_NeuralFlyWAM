"""Render complete accepted confirmation panels, without pooling the two cohorts."""

# ruff: noqa: E402 -- standalone access to the independent statistics verifier.
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from round18_verify_compact import verify

LABELS = {"log": "Re-simulated", "mixed": "Mixed"}
CELLS = [f"{r}_{b}" for r in LABELS for b in ["short", "long"]]


def interval(value, digits=3, columns=False):
    lo, hi = value["descriptive_t95ci"]
    # Preserve the sign of nonzero endpoints when ordinary rounding hides it.
    while any(x != 0 and float(f"{x:.{digits}f}") == 0 for x in [lo, hi]):
        digits += 1
    separator = "$ & $" if columns else r"\;"
    return f"${value['mean']:+.{digits}f}{separator}[{lo:.{digits}f},{hi:.{digits}f}]$"


def contrast_label(key):
    if key == "budget_change_in_coverage_interaction":
        return r"L--S change in Mixed--Re-simulated $\Delta$(U--F)"
    tokens = key.split("/")
    if tokens[0] in ["short", "long"]:
        return (
            ("S" if tokens[0] == "short" else "L")
            + ": Mixed--Re-simulated, "
            + (r"$\Delta$(U--F)" if tokens[2] == "change_in_U-F" else tokens[2].title())
        )
    assert tokens[0] in LABELS and tokens[1] == "long-short"
    return (
        LABELS[tokens[0]]
        + ": L--S, "
        + (r"$\Delta$(U--F)" if tokens[2] == "change_in_U-F" else tokens[2].title())
    )


def main():
    panels = {
        panel: json.loads((ROOT / f"reports/ROUND18_{panel}_RESULTS.json").read_text())
        for panel in ["CONTROL", "QUERY", "OFFLINE"]
    }
    for panel in panels.values():
        verify(panel)
    original = json.loads((ROOT / "reports/ROUND16_CONTROL_RESULTS.json").read_text())["summary"][
        "rmse_m"
    ]
    control = panels["CONTROL"]["summary"]["rmse_m"]
    lines = [
        r"\begin{table}[t]\centering\small",
        r"\caption{Long-budget Mixed minus Re-simulated tracking RMSE (m). Each cohort contains five independently generated training datasets and uses its own shared evaluation roster. Cohorts are analyzed separately. Values are means with descriptive paired 95\% intervals; negative values favor Mixed.}\label{tab:confirmation_main}",
        r"\setlength{\tabcolsep}{4pt}\begin{tabular}{llrr}\toprule",
        r"Wind (m/s) & Memory design & Original datasets & Confirmation datasets\\\midrule",
    ]
    for wind in ["6.1", "8.5"]:
        for arm in ["frozen", "updated"]:
            key = "long/mixed-log/" + arm
            lines.append(
                f"{wind} & {arm.title()} & "
                + " & ".join(interval(data[wind]["contrasts"][key]) for data in [original, control])
                + r"\\"
            )
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    (ROOT / "paper/round18_confirmation_main_table.tex").write_text("\n".join(lines) + "\n")

    out = []
    specifications = [
        ("CONTROL", "rmse_m", "Failure-aware tracking RMSE (m)", ["0.0", "4.9", "6.1", "8.5"], 4),
        (
            "QUERY",
            "physical_solution_cost",
            "Physical solution cost",
            ["0.0", "4.9", "6.1", "8.5"],
            3,
        ),
        (
            "QUERY",
            "query_E",
            "Prediction error $E$ on common candidate banks",
            ["0.0", "4.9", "6.1", "8.5"],
            4,
        ),
        (
            "QUERY",
            "position_m",
            "Physical position error (m) on common candidate banks",
            ["0.0", "4.9", "6.1", "8.5"],
            4,
        ),
        (
            "OFFLINE",
            "original_E",
            "Prediction error $E$ on static logged-action tests",
            ["3.7", "6.1", "8.5"],
            4,
        ),
    ]
    for panel, metric, title, winds, digits in specifications:
        data = panels[panel]["summary"] if panel == "OFFLINE" else panels[panel]["summary"][metric]
        out += [
            r"\begin{table}[!htbp]\centering\footnotesize",
            r"\caption{Independent confirmation: "
            + title
            + r". All 120 models retained; S/L denotes the short/long budget. Means and paired U--F contrasts use five new training datasets. Intervals are descriptive and nonsimultaneous; extra digits preserve near-zero endpoint signs.}",
            r"\begin{tabular}{llrrrr}\toprule",
            r"Wind & Recipe / budget & Frozen & Updated & U--F & 95\% interval\\\midrule",
        ]
        for wi, wind in enumerate(winds):
            for cell in CELLS:
                recipe, budget = cell.split("_")
                label = LABELS[recipe] + " / " + ("S" if budget == "short" else "L")
                frozen, updated = [
                    data[wind]["arm_means"][cell + "/" + a] for a in ["frozen", "updated"]
                ]
                out.append(
                    f"{float(wind):g} & {label} & {frozen:.{digits}f} & {updated:.{digits}f} & "
                    + interval(data[wind]["contrasts"][cell + "/U-F"], digits, columns=True)
                    + r"\\"
                )
            if wi < len(winds) - 1:
                out.append(r"\midrule")
        out += [r"\bottomrule\end{tabular}\end{table}"]

    for wind in ["0.0", "4.9", "6.1", "8.5"]:
        out += [
            r"\begin{table}[!htbp]\centering\footnotesize",
            r"\caption{Independent confirmation: complete recipe and budget contrasts in tracking RMSE (m) at "
            + f"{float(wind):g}"
            + r"\,m/s. Each contrast uses five new paired training datasets; intervals are descriptive and nonsimultaneous. U--F contrasts appear in the preceding mean table.}",
            r"\begin{tabular}{lrrr}\toprule",
            r"Contrast & Mean & 95\% interval & Negative / 5\\\midrule",
        ]
        for key, value in control[wind]["contrasts"].items():
            if key.endswith("/U-F"):
                continue
            out.append(
                contrast_label(key)
                + " & "
                + interval(value, 4, columns=True)
                + f" & {value['negative_datasets']}/5"
                + r"\\"
            )
        out += [r"\bottomrule\end{tabular}\end{table}"]
    (ROOT / "paper/round18_confirmation_tables.tex").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
