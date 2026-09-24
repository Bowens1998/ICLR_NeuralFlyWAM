"""Apply the prewritten confirmation interpretation rule to complete control data."""

import argparse
import hashlib
import json
from pathlib import Path

from round18_verify_compact import verify


def classify(control):
    primary = []
    for wind in ["6.1", "8.5"]:
        for arm, sign in [("frozen", 1), ("updated", -1)]:
            value = control[wind]["contrasts"]["long/mixed-log/" + arm]
            lower, upper = value["descriptive_t95ci"]
            primary.append(
                dict(
                    wind=float(wind),
                    memory_design=arm.title(),
                    contrast="long-budget Mixed minus Re-simulated tracking RMSE (m)",
                    expected_direction="positive" if sign == 1 else "negative",
                    mean_sign_recurs=sign * value["mean"] > 0,
                    interval_excludes_zero_in_expected_direction=(
                        lower > 0 if sign == 1 else upper < 0
                    ),
                    **value,
                )
            )
    signs = all(x["mean_sign_recurs"] for x in primary)
    precision = all(x["interval_excludes_zero_in_expected_direction"] for x in primary)
    classification = (
        "precise_opposed_effects_confirmation"
        if signs and precision
        else "directional_recurrence_with_unresolved_precision"
        if signs
        else "full_opposed_effects_pattern_not_reproduced"
    )
    return dict(
        classification=classification,
        all_four_expected_mean_signs=signs,
        all_four_expected_zero_excluding_intervals=precision,
        primary_within_design_contrasts=primary,
        primary_interactions={
            wind: control[wind]["contrasts"]["long/mixed-log/change_in_U-F"]
            for wind in ["6.1", "8.5"]
        },
        short_budget_primary_wind_contrasts={
            wind: {
                arm: control[wind]["contrasts"]["short/mixed-log/" + arm]
                for arm in ["frozen", "updated"]
            }
            for wind in ["6.1", "8.5"]
        },
        interpretation="Protocol-defined descriptive rule across five new training datasets, conditional on the shared new evaluation roster. Intervals are nonsimultaneous; this is not a multiplicity-adjusted hypothesis test. Original and new cohorts remain separate. Same physics family; no hardware-flight or unique-mechanism claim.",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    data = json.loads(a.input.read_text())
    assert data["panel"] == "control"
    verification = verify(data)
    result = classify(data["summary"]["rmse_m"])
    result["input_sha256"] = hashlib.sha256(a.input.read_bytes()).hexdigest()
    result["independent_compact_verification"] = verification
    with a.output.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(result["classification"])


if __name__ == "__main__":
    main()
