"""Coverage appendix tables from accepted complete records only."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
q = json.loads((ROOT / "reports/ROUND15_QUERY_RESULTS.json").read_text())
c = json.loads((ROOT / "reports/ROUND15_CONTROL_RESULTS.json").read_text())
o = json.loads((ROOT / "reports/ROUND15_OFFLINE_RESULTS.json").read_text())
assert q["accepted_records"] == c["accepted_records"] == 1800
panels = [
    ("Query $E$", q["summary"]["query_E"]),
    ("Physical solution cost", q["summary"]["physical_solution_cost"]),
    ("Query position (m)", q["summary"]["position_m"]),
    ("Tracking RMSE (m)", c["summary"]["rmse_m"]),
    ("Logged $E$", o["summary"]),
]
lines = [
    r"\begin{table}[t]\centering\small",
    r"\caption{Action-coverage factorial: all four absolute means. Three seeds and ten evaluation episodes/anchors are averaged within each of five training datasets; logged errors average original test windows. Lower is better.}\label{tab:coverage_means}",
    r"\begin{tabular}{llrrrr}\toprule",
    r"Outcome & Wind & Re-simulated F & Re-simulated U & Mixed F & Mixed U\\\midrule",
]
for label, ww in panels:
    for w, r in ww.items():
        m = r["arm_means"]
        lines.append(
            label
            + " & "
            + w
            + " & "
            + " & ".join(
                f"{m[k]:.5f}"
                for k in ["log/frozen", "log/updated", "mixed/frozen", "mixed/updated"]
            )
            + r"\\"
        )
lines += [
    r"\bottomrule\end{tabular}\end{table}",
    r"\begin{table}[t]\centering\scriptsize",
    r"\caption{All memory gaps and coverage interactions. Brackets are descriptive paired 95\% intervals across five datasets, not simultaneous intervals. Mixed intervals crossing zero do not establish equivalence.}\label{tab:coverage_contrasts}",
    r"\begin{tabular}{llrrr}\toprule",
    r"Outcome & Wind & Re-simulated U--F [CI] & Mixed U--F [CI] & Interaction [CI]\\\midrule",
]
for label, ww in panels:
    for w, r in ww.items():
        v = []
        for k in [
            "log_updated_minus_frozen",
            "mixed_updated_minus_frozen",
            "coverage_by_memory_interaction",
        ]:
            d = r["contrasts"][k]
            lo, hi = d["descriptive_t95ci"]
            v.append(f"{d['mean']:+.4f} [{lo:+.4f},{hi:+.4f}]")
        lines.append(label + " & " + w + " & " + " & ".join(v) + r"\\")
lines += [
    r"\bottomrule\end{tabular}\end{table}",
    r"\begin{table}[t]\centering\small",
    r"\caption{Every training-dataset coverage interaction, in original dataset order. Negative values indicate that Mixed training reduces the Updated minus Frozen gap.}\label{tab:coverage_datasets}",
    r"\begin{tabular}{llrrrrr}\toprule",
    r"Outcome & Wind & D1 & D2 & D3 & D4 & D5\\\midrule",
]
for label, ww in panels:
    for w, r in ww.items():
        lines.append(
            label
            + " & "
            + w
            + " & "
            + " & ".join(
                f"{v:+.4f}"
                for v in r["contrasts"]["coverage_by_memory_interaction"]["dataset_differences"]
            )
            + r"\\"
        )
lines += [r"\bottomrule\end{tabular}\end{table}"]
(ROOT / "paper/round15_coverage_tables.tex").write_text("\n".join(lines) + "\n")
main_lines=[r'\begin{table}[t]\centering\small',r'\caption{Tracking RMSE (m) in the matched action-coverage experiment. These are newly trained models; even Re-simulated training changes the original mean ordering. Each mean uses all five datasets, three seeds and ten episodes.}\label{tab:coverage_main}',r'\begin{tabular}{lrrrr}\toprule',r'Wind (m/s) & Re-simulated Frozen & Re-simulated Updated & Mixed Frozen & Mixed Updated\\\midrule']
for w,r in c['summary']['rmse_m'].items():
 main_lines.append(w+' & '+' & '.join(f"{r['arm_means'][k]:.3f}" for k in ['log/frozen','log/updated','mixed/frozen','mixed/updated'])+r'\\')
main_lines += [r'\bottomrule\end{tabular}\end{table}']
(ROOT/'paper/round15_coverage_main_table.tex').write_text('\n'.join(main_lines)+'\n')
