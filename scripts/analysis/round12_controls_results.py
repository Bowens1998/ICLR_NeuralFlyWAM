"""Full-roster dataset-level Round12 control summaries, with unique baseline reuse."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/analysis"))
sys.path.insert(0, str(REPO / "scripts/sim"))

import round11d_validate as legacy_validation  # noqa: E402
import round12_control_protocol as protocol  # noqa: E402
import round12_controls_accept as acceptance  # noqa: E402

ARMS = ["frozen_r0", "frozen_r1", "updated_r0", "updated_r1"]
WINDS = [0., 4.9, 6.1, 8.5]
METRICS = ["capped_rmse_m", "failure_rate"]
DIAGNOSTICS = ["throttle_deviation_rms", "command_tilt_rms_rad", "planning_mean_seconds",
               "planning_p95_seconds", "elapsed_seconds"]
require = acceptance.require


def describe(values):
    array = np.asarray(values, dtype=float)
    require(array.shape == (5,) and np.isfinite(array).all(), "five finite dataset values required")
    mean = float(array.mean())
    se = float(array.std(ddof=1) / np.sqrt(5))
    interval = [float(x) for x in t.interval(.95, 4, loc=mean, scale=se)] if se else [mean, mean]
    return dict(dataset_values=array.tolist(), mean=mean, descriptive_t95ci=interval,
                positive_datasets=int((array > 0).sum()), negative_datasets=int((array < 0).sum()),
                zero_datasets=int((array == 0).sum()), n_datasets=5)


def metric(record, name):
    value = record["capped_tracking_rmse_m"] if name == "capped_rmse_m" else float(record["failed"])
    require(np.isfinite(value), "nonfinite primary outcome")
    return float(value)


def arm(entry):
    return ("updated" if entry["update_context"] else "frozen") + "_r" + str(int(entry["context_norm"] == "ln"))


def diagnostic_summary(records):
    summary = {}
    for name in DIAGNOSTICS:
        available = [record[name] for record in records if record.get(name) is not None]
        require(np.isfinite(available).all(), "nonfinite observed diagnostic")
        summary[name] = dict(mean=float(np.mean(available)) if available else None,
                             observed_episodes=len(available), total_episodes=len(records))
    return summary


def load_baseline(roster, root, acceptance_path):
    """Validate each referenced original artifact against the locked accepted SHA."""
    root = Path(root)
    baseline = roster["baseline"]
    require(protocol.file_hash(acceptance_path) == baseline["acceptance_sha256"],
            "baseline acceptance SHA mismatch")
    accepted = json.loads(Path(acceptance_path).read_text())
    require(accepted["group"] == "all" and accepted["accepted_episodes"] == 2560
            and accepted["cross_group_warmup_check"] is True,
            "baseline does not have complete original acceptance")
    require(accepted["source"] == baseline["source_commit"] == protocol.BASELINE_SOURCE
            and accepted["roster_sha256"] == baseline["roster_sha256"], "baseline source mismatch")
    files = accepted["files"]
    require(len(files) == 2560 and {item["index"] for item in files} == set(range(2560)),
            "invalid baseline accepted roster")
    hashes = {item["index"]: item["sha256"] for item in files}
    locked_hashes = {int(key): value for key, value in baseline["episode_sha256"].items()}
    require(hashes == locked_hashes, "baseline episode hash contract mismatch")
    originals = protocol.previous.design()["rows"]
    manifests = roster["checkpoint_manifest"]["checkpoints"]
    identities = {(entry["model_name"], entry["model_seed"]): entry for entry in manifests}
    records, warmups = [], {}
    for reuse in roster["baseline_reuse"]:
        index = reuse["index"]
        expected_row = originals[index]
        require(reuse["sha256"] == hashes[index], "baseline reuse SHA mismatch")
        path = root / f"episode_{index:04d}.json"
        require(protocol.file_hash(path) == hashes[index], "baseline episode bytes changed")
        result = json.loads(path.read_text())
        claim = json.loads(path.with_suffix(".claim").read_text())
        checkpoint_sha = None
        if expected_row["kind"] == "model":
            require(roster["checkpoint_manifest"].get("reuse_round11_learned") is True,
                    "another family cannot reuse original learned outcomes")
            checkpoint_sha = identities[(expected_row["model"], expected_row["model_seed"])]["checkpoint_sha256"]
        legacy_validation.validate_episode(result, claim, expected_row, baseline["roster_sha256"],
                                           checkpoint_sha)
        # Full original acceptance already establishes all cross-controller warmups;
        # still compare the retained subset to guard against accidental recombination.
        key = (expected_row["wind"], expected_row["episode"])
        current = dict(warmup_state=result["warmup_state"], actions=result["actions"][:100])
        if key in warmups:
            require(acceptance.same_warmup(warmups[key], current), "baseline common warmup mismatch")
        if (key not in warmups or len(current["actions"]) > len(warmups[key]["actions"])
                or (warmups[key]["warmup_state"] is None and current["warmup_state"] is not None)):
            warmups[key] = current
        record = acceptance.compact(result)
        record["row"] = dict(expected_row, setting="baseline")
        record["origin"] = dict(study="round11", index=index, sha256=hashes[index])
        records.append(record)
    require(len(records) == len(roster["baseline_reuse"]), "incomplete baseline reuse")
    return records


def effect_arrays(data, nominal):
    frozen = data["frozen_r1"] - data["frozen_r0"]
    updated = data["updated_r1"] - data["updated_r0"]
    effects = dict(frozen_recurrence_LN_effect=frozen, updated_recurrence_LN_effect=updated,
                   recurrence_effect_updated_minus_frozen=updated-frozen,
                   updated_minus_frozen_no_norm=data["updated_r0"]-data["frozen_r0"],
                   updated_minus_frozen_LN=data["updated_r1"]-data["frozen_r1"])
    effects.update({f"{name}_minus_nominal": values-nominal for name, values in data.items()})
    return effects


def aggregate(roster, new_records, baseline_records):
    require(roster["phase"] == "formal", "pilots cannot enter scientific aggregation")
    require(len(new_records) == len(roster["rows"]), "incomplete new episode roster")
    indices = [record["row"]["index"] for record in new_records]
    require(set(indices) == set(range(len(roster["rows"]))) and len(set(indices)) == len(indices),
            "duplicate or missing new episode index")
    for record in new_records:
        require(record["row"] == roster["rows"][record["row"]["index"]], "new episode identity mismatch")
    expected_reuse = {entry["index"] for entry in roster["baseline_reuse"]}
    require(len(baseline_records) == len(expected_reuse)
            and {record["row"]["index"] for record in baseline_records} == expected_reuse,
            "duplicate or missing reused baseline episode")
    original_rows = protocol.previous.design()["rows"]
    for record in baseline_records:
        require(record["row"] == dict(original_rows[record["row"]["index"]],
                                      setting="baseline"), "reused baseline row mismatch")
    manifest = roster["checkpoint_manifest"]
    entries = {(entry["model_name"], entry["model_seed"]): entry for entry in manifest["checkpoints"]}
    seeds = set(manifest["model_seeds"])
    records = list(new_records) + list(baseline_records)
    learned, refs = defaultdict(list), defaultdict(list)
    for record in records:
        row = record["row"]
        require(type(record["failed"]) is bool, "non-Boolean failure observation")
        require(0 <= metric(record, "capped_rmse_m") <= 5., "invalid capped RMSE")
        if row["kind"] == "model":
            entry = entries[(row["model"], row["model_seed"])]
            require(entry["dataset_replicate"] == row["dataset_replicate"], "dataset identity mismatch")
            learned[(row["setting"], row["wind"], row["dataset_replicate"], arm(entry))].append(record)
        else:
            refs[(row["setting"], row["wind"], row["kind"])].append(record)
    settings = [name for name in protocol.SETTINGS if any(key[0] == name for key in learned)]
    require("baseline" in settings, "paired sensitivity requires a learned baseline")
    reference_pool, summaries, raw_effects = {}, {}, {}
    for (setting, wind, kind), reference_records in sorted(refs.items()):
        require(len(reference_records) == 10
                and {record["row"]["episode"] for record in reference_records} == set(range(10)),
                "exactly ten unique reference episodes required")
        key = f"{setting}:{wind}:{kind}"
        reference_pool[key] = dict(source_setting=setting, wind=wind, controller_kind=kind,
                                   unique_episodes=10, outcomes={name: float(np.mean(
                                       [metric(record, name) for record in reference_records])) for name in METRICS},
                                   diagnostics=diagnostic_summary(reference_records))
    for setting in settings:
        summaries[setting] = dict(specification=protocol.SETTINGS[setting], winds={})
        for wind in WINDS:
            reference_keys = {}
            for kind in protocol.previous.KINDS[1:]:
                source_setting = setting
                if (setting == "baseline"
                        or (protocol.SETTINGS[setting]["study"] == "B" and kind in ("pd", "preview_pd"))):
                    source_setting = "baseline"
                key = f"{source_setting}:{wind}:{kind}"
                require(key in reference_pool, "missing required unique reference group")
                reference_keys[kind] = key
            data = {name: {name_arm: [] for name_arm in ARMS} for name in METRICS}
            diagnostics = {name_arm: [] for name_arm in ARMS}
            for dataset in range(5):
                for name_arm in ARMS:
                    group = learned[(setting, wind, dataset, name_arm)]
                    keys = {(record["row"]["model_seed"], record["row"]["episode"]) for record in group}
                    require(len(group) == 30 and keys == {(seed, ep) for seed in seeds for ep in range(10)},
                            "full three-model-seed by ten-episode cross required")
                    for name in METRICS:
                        data[name][name_arm].append(float(np.mean([metric(record, name) for record in group])))
                    diagnostics[name_arm].append(dict(dataset_replicate=dataset, **diagnostic_summary(group)))
            contrasts = {}
            for name in METRICS:
                arrays = {name_arm: np.asarray(values) for name_arm, values in data[name].items()}
                nominal = reference_pool[reference_keys["nominal"]]["outcomes"][name]
                effects = effect_arrays(arrays, nominal)
                raw_effects[(setting, wind, name)] = effects
                contrasts[name] = {key: describe(value) for key, value in effects.items()}
            summaries[setting]["winds"][str(wind)] = dict(
                reference_keys=reference_keys, dataset_scores=data,
                arm_summaries={name: {key: describe(value) for key, value in values.items()}
                               for name, values in data.items()}, contrasts=contrasts,
                diagnostics_by_dataset=diagnostics)
    for setting in settings:
        for wind in WINDS:
            changes = {}
            if setting != "baseline":
                for name in METRICS:
                    changes[name] = {key: describe(value-raw_effects[("baseline", wind, name)][key])
                                     for key, value in raw_effects[(setting, wind, name)].items()}
            summaries[setting]["winds"][str(wind)]["contrasts_minus_baseline"] = changes
    return dict(schema="round12-control-results-v1", family_id=manifest["family_id"],
                model_seeds=manifest["model_seeds"], reference_pool=reference_pool, settings=summaries,
                counts=dict(new_episodes=len(new_records), unique_reused_episodes=len(baseline_records),
                            unique_total_episodes=len(records), unique_reference_episodes=sum(len(group) for group in refs.values()),
                            retained_failures=sum(record["failed"] for record in records)),
                caveat="Descriptive paired t95 intervals use five training datasets (df=4), conditional on "
                       "the same ten evaluation episodes. Average three model seeds and ten episodes "
                       "within dataset before contrasts. No p-values, simultaneous intervals, equivalence "
                       "claims or physics-population inference. References and baseline episodes remain "
                       "unique; reference_keys explicitly identify reuse. Diagnostics average observed "
                       "values only and report availability; failed episodes retain capped primary outcomes.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--round11-root", type=Path, required=True)
    parser.add_argument("--round11-acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roster, records, accepted = acceptance.load_accepted(args.protocol, args.root, args.acceptance)
    baseline = load_baseline(roster, args.round11_root, args.round11_acceptance)
    report = aggregate(roster, records, baseline)
    report.update(protocol_sha256=protocol.file_hash(args.protocol),
                  acceptance_sha256=protocol.file_hash(args.acceptance),
                  baseline_acceptance_sha256=protocol.file_hash(args.round11_acceptance),
                  execution_source_commit=accepted["source_commit"],
                  analysis_source_sha256={Path(__file__).name: protocol.file_hash(Path(__file__)),
                                          "round12_controls_accept.py": protocol.file_hash(Path(acceptance.__file__))},
                  episode_diagnostics=records,
                  baseline_episode_diagnostics=baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["counts"]))


if __name__ == "__main__":
    main()
