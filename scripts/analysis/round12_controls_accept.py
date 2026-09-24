"""Strict artifact acceptance and timing summaries for Round12 control episodes."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/sim"))

import round12_control_protocol as protocol  # noqa: E402
import round12_controls as executor  # noqa: E402

ACCEPT_SCHEMA = "round12-control-acceptance-v1"
PREDICTION_FAILURES = {"nonfinite_candidate_action", "nonfinite_prediction"}
STATE_FAILURES = {"nonfinite_state", "tracking_error", "position_norm", "velocity_norm",
                  "body_rate_norm"}
METADATA_KEYS = {"schema", "phase", "family_id", "row", "source_commit", "protocol_sha256",
                 "source_sha256", "checkpoint_sha256", "checkpoint_validation", "started_utc"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_number(actual, expected, label):
    if expected is None:
        require(actual is None, f"{label}: expected null")
    elif isinstance(expected, bool):
        require(type(actual) is bool and actual == expected, f"{label}: invalid Boolean")
    elif isinstance(expected, int):
        require(type(actual) is int and actual == expected, f"{label}: invalid count")
    else:
        require(isinstance(actual, int | float) and not isinstance(actual, bool)
                and np.isfinite(actual) and np.isclose(actual, expected, rtol=1e-10, atol=1e-12),
                f"{label}: numeric mismatch")


def json_equal(left, right):
    # Parameters contain tuples in memory but lists after JSON serialization.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False)


def scope_rows(roster, kind):
    require(kind in ("all", "gpu", "cpu"), "invalid acceptance scope")
    return [row for row in roster["rows"] if kind == "all"
            or (row["kind"] == "model") == (kind == "gpu")]


def checkpoint_identity(roster, row):
    if row["kind"] != "model":
        return None
    entry = roster["checkpoint_manifest"]["checkpoints"][row["checkpoint_index"]]
    return dict(checkpoint_sha256=entry["checkpoint_sha256"],
                provenance=dict(git_commit=entry["source_commit"], model_name=entry["model_name"],
                                seed=entry["model_seed"], config_hash=entry["config_hash"],
                                split_checksum=entry["split_checksum"],
                                normalizer_sha256=entry["normalizer_sha256"]),
                config_sha256=entry["config_sha256"], data_artifacts_checked=True)


def validate_episode(result, claim, row, roster, protocol_sha256, source):
    """Validate one successful OR numerically failed controller episode."""
    r = result
    require(re.fullmatch(r"[0-9a-f]{40}", source) is not None, "invalid execution source")
    identity = checkpoint_identity(roster, row)
    fixed = dict(schema=protocol.SCHEMA, phase=roster["phase"], family_id=roster["family_id"],
                 row=row, source_commit=source, protocol_sha256=protocol_sha256,
                 source_sha256=roster["source_sha256"], checkpoint_validation=identity,
                 checkpoint_sha256=identity["checkpoint_sha256"] if identity else None)
    require(set(claim) == METADATA_KEYS, "claim metadata fields differ from executor contract")
    for key, value in fixed.items():
        require(json_equal(r.get(key), value) and json_equal(claim.get(key), value),
                f"identity mismatch: {key}")
    require(r.get("started_utc") == claim["started_utc"], "claim/result start time mismatch")
    start = datetime.fromisoformat(r["started_utc"])
    require(start.tzinfo is not None, "execution timestamp requires timezone")
    plant, design = executor.parameter_pair(row["plant_changes"])
    require(json_equal(r["plant_parameters"], executor.asdict(plant)), "plant parameter mismatch")
    require(json_equal(r["controller_parameters"], executor.asdict(design)),
            "controller design parameter mismatch")
    expected_device = "cuda" if row["kind"] == "model" else "cpu"
    require(isinstance(r["device"], str)
            and re.fullmatch(expected_device + r"(?::[0-9]+)?", r["device"]) is not None,
            "device does not match GPU-model/CPU-reference routing")
    require(all(isinstance(r.get(key), str) and r[key]
                for key in ("torch_version", "numpy_version")), "missing runtime versions")
    steps = roster["scored_steps"]
    require(type(r["failed"]) is bool, "failure flag must be Boolean")
    require(isinstance(r["errors_m"], list), "errors must be a list")
    errors = r["errors_m"]
    require(all(value is None or (isinstance(value, int | float) and not isinstance(value, bool)
                                 and np.isfinite(value) and value >= 0) for value in errors),
            "invalid recorded error")
    missing = [i for i, value in enumerate(errors) if value is None]
    require(not missing or (r["failed"] and missing == [len(errors)-1]),
            "nonfinite error must terminate the episode")
    expected = executor.scoring.score([np.nan if value is None else value for value in errors],
                                     r["failed"], steps=steps)
    for key, value in expected.items():
        same_number(r[key], value, key)
    actions = np.asarray(r["actions"], dtype=float)
    if actions.size == 0:
        require(r["actions"] == [], "empty actions must use an empty list")
        actions = np.empty((0, 5))
    require(actions.ndim == 2 and actions.shape[1] == 5 and np.isfinite(actions).all(),
            "invalid action shape or nonfinite executed action")
    require(np.all((actions[:, 0] >= .05-1e-12) & (actions[:, 0] <= 1.+1e-12)),
            "executed throttle outside command bounds")
    require(np.allclose(np.linalg.norm(actions[:, 1:], axis=1), 1., atol=1e-6, rtol=0),
            "executed action quaternion is not unit norm")
    latency = np.asarray(r["planning_seconds"], dtype=float)
    require(latency.ndim == 1 and np.isfinite(latency).all() and np.all(latency > 0),
            "invalid latency sequence")
    reason, failure_step = r["failure_reason"], r["failure_step"]
    if r["failed"]:
        require(reason in STATE_FAILURES | PREDICTION_FAILURES | {"nonfinite_action"},
                "unrecognized numerical controller failure")
        require(type(failure_step) is int and 0 <= failure_step < 100+steps,
                "invalid failure step")
        after_action = reason in STATE_FAILURES
        require(len(actions) == failure_step + int(after_action), "failure action count mismatch")
        require(len(errors) == max(0, len(actions)-100), "failure error count mismatch")
        expected_latencies = max(0, len(actions)-100)
        if reason == "nonfinite_action" and failure_step >= 100:
            expected_latencies += 1
        if reason in PREDICTION_FAILURES:
            require(row["kind"] in ("model", "nominal", "true_mean") and failure_step >= 100,
                    "prediction failure outside planning")
        reached_planning = failure_step >= 100
    else:
        require(reason is None and failure_step is None, "successful episode has failure metadata")
        require(len(actions) == 100+steps and len(errors) == steps, "incomplete successful episode")
        expected_latencies, reached_planning = steps, True
    require(len(latency) == (0 if row["kind"] == "pd" else expected_latencies),
            "planning latency count mismatch")
    same_number(r["observed_effort_steps"], len(errors), "observed_effort_steps")
    require(len(actions) <= 100+steps, "too many executed actions")
    same_number(r["planning_mean_seconds"], float(latency.mean()) if len(latency) else None,
                "planning_mean_seconds")
    same_number(r["planning_p95_seconds"], float(np.quantile(latency, .95)) if len(latency) else None,
                "planning_p95_seconds")
    require(np.isfinite(r["elapsed_seconds"]) and r["elapsed_seconds"] > float(latency.sum()),
            "invalid episode wall time")
    if reached_planning:
        warmup = r["warmup_state"]
        require(isinstance(warmup, dict) and set(warmup) == {"p", "v", "w", "R"},
                "missing warmup state")
        for key, value in warmup.items():
            array = np.asarray(value, dtype=float)
            require(array.shape == ((3, 3) if key == "R" else (3,)) and np.isfinite(array).all(),
                    "invalid warmup state")
        rotation = np.asarray(warmup["R"])
        require(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
                and np.isclose(np.linalg.det(rotation), 1., atol=1e-8, rtol=0),
                "invalid warmup rotation")
        require(np.isfinite(r["warmup_seconds"])
                and 0 < r["warmup_seconds"] < r["elapsed_seconds"], "invalid warmup time")
    else:
        require(r["warmup_state"] is None and r["warmup_seconds"] is None,
                "warmup failure claims completed warmup")
    if len(errors):
        q = actions[100:, 1:]
        tilt = np.arccos(np.clip(1-2*(q[:, 0]**2+q[:, 1]**2), -1., 1.))
        same_number(r["command_tilt_rms_rad"], float(np.sqrt(np.mean(tilt**2))), "tilt effort")
        same_number(r["throttle_deviation_rms"], float(np.sqrt(np.mean((actions[100:, 0]-.3924)**2))),
                    "throttle effort")
    else:
        require(r["command_tilt_rms_rad"] is None and r["throttle_deviation_rms"] is None,
                "unobserved effort must be null")
    return True


def compact(result):
    keys = ("row", "failed", "failure_reason", "failure_step", "capped_tracking_rmse_m",
            "successful_tracking_rmse_m", "observed_steps", "scored_steps", "observed_effort_steps",
            "throttle_deviation_rms", "command_tilt_rms_rad", "planning_mean_seconds",
            "planning_p95_seconds", "elapsed_seconds", "warmup_seconds", "device")
    return {key: result[key] for key in keys}


def update_warmup_checks(previous, result):
    row = result["row"]
    key = (protocol.object_hash(result["plant_parameters"]), row["wind"], row["episode"])
    current = dict(warmup_state=result["warmup_state"], actions=result["actions"][:100])
    if key in previous:
        require(same_warmup(previous[key], current), "shared-plant common warmup identity mismatch")
    if (key not in previous or len(current["actions"]) > len(previous[key]["actions"])
            or (previous[key]["warmup_state"] is None and current["warmup_state"] is not None)):
        previous[key] = current


def same_warmup(left, right):
    """Compare available prefixes; do not invalidate a retained early warmup failure."""
    la, ra = np.asarray(left["actions"]).reshape(-1, 5), np.asarray(right["actions"]).reshape(-1, 5)
    shared = min(len(la), len(ra))
    if not np.allclose(la[:shared], ra[:shared], atol=1e-12, rtol=0):
        return False
    ls, rs = left["warmup_state"], right["warmup_state"]
    if ls is None or rs is None:
        return True
    return set(ls) == set(rs) and all(np.asarray(ls[key]).shape == np.asarray(rs[key]).shape
                                     and np.allclose(ls[key], rs[key], atol=1e-12, rtol=0) for key in ls)


def timing_projection(roster, records, kind="all", safety_factor=2., startup_seconds=300.,
                      chunk_size=1):
    require(safety_factor >= 1 and startup_seconds >= 0 and type(chunk_size) is int and chunk_size > 0,
            "invalid timing projection settings")
    formal = protocol.design(roster["checkpoint_manifest"], roster["baseline"], phase="formal",
                             studies=roster["studies"], sources=roster["source_sha256"],
                             evaluation_panel=roster.get("evaluation_panel"))
    counts = defaultdict(int)
    for row in scope_rows(formal, kind):
        counts[(row["setting"], row["kind"])] += 1
    measured = defaultdict(list)
    for record in records:
        measured[(record["row"]["setting"], record["row"]["kind"])].append(record)
    groups = []
    total_seconds = 0.
    complete = True
    for (setting, controller), count in counts.items():
        available = measured[(setting, controller)]
        completed = [r for r in available if r["observed_steps"] == r["scored_steps"]
                     and (not r["failed"] or r["scored_steps"] == 1500)]
        wall = [r["warmup_seconds"] + (r["elapsed_seconds"]-r["warmup_seconds"])
                * 1500/r["scored_steps"] for r in completed]
        planning = [(r["planning_mean_seconds"] or 0.) * 1500 for r in completed]
        estimate = max(wall) if wall else None
        complete &= estimate is not None
        if estimate is not None:
            total_seconds += count * estimate
        groups.append(dict(setting=setting, controller_kind=controller, formal_episodes=count,
                           measured_episodes=len(available), retained_failures=sum(r["failed"] for r in available),
                           complete_timing_episodes=len(completed),
                           full_length_timing_episodes=sum(r["scored_steps"] == 1500 for r in completed),
                           max_observed_wall_seconds=max((r["elapsed_seconds"] for r in available), default=None),
                           max_projected_full_wall_seconds=estimate,
                           max_projected_planning_seconds=max(planning, default=None),
                           conservative_chunk_seconds=(safety_factor*estimate*chunk_size+startup_seconds
                                                       if estimate is not None else None),
                           projected_group_compute_hours=count*estimate/3600 if estimate is not None else None))
    return dict(groups=groups, resource_projection_complete=bool(complete),
                projected_total_compute_hours=total_seconds/3600 if complete else None,
                safety_factor=safety_factor, startup_seconds_per_chunk=startup_seconds,
                chunk_size=chunk_size,
                caveat="Wall time excludes checkpoint loading and process startup; startup allowance is explicit. "
                       "Short pilots use linear extrapolation; full-length pilots are preferable. "
                       "Numerical failures remain accepted data; incomplete failed runs cannot alone "
                       "certify full-run timing. Failures on the final scored step retain full timing. "
                       "These are measured projections, not scheduler allocation guarantees.")


def accept(protocol_path, root, source, kind="all", *, safety_factor=2., startup_seconds=300.,
           chunk_size=1):
    roster = json.loads(Path(protocol_path).read_text())
    protocol.validate_protocol(roster)
    digest = protocol.file_hash(protocol_path)
    root = Path(root)
    selected = scope_rows(roster, kind)
    require(bool(selected), "selected scope contains no episodes")
    for path in root.glob("episode_*.json"):
        match = re.fullmatch(r"episode_([0-9]{5})\.json", path.name)
        require(match is not None and int(match[1]) < len(roster["rows"]),
                "unexpected episode filename or index")
    files, summary, warmups = [], [], {}
    for row in selected:
        path = root / f"episode_{row['index']:05d}.json"
        require(not path.with_suffix(".error").exists(), f"software failure artifact at index {row['index']}")
        result = json.loads(path.read_text())
        claim = json.loads(path.with_suffix(".claim").read_text())
        validate_episode(result, claim, row, roster, digest, source)
        update_warmup_checks(warmups, result)
        files.append(dict(index=row["index"], sha256=protocol.file_hash(path),
                          claim_sha256=protocol.file_hash(path.with_suffix(".claim"))))
        summary.append(compact(result))
    report = dict(schema=ACCEPT_SCHEMA, phase=roster["phase"], family_id=roster["family_id"], kind=kind,
                  source_commit=source, protocol_sha256=digest, source_sha256=roster["source_sha256"],
                  accepted_episodes=len(files), failed_episodes_retained=sum(r["failed"] for r in summary),
                  complete_roster=kind == "all", cross_group_warmup_check=kind == "all",
                  warmup_groups=len(warmups), files=files, primary_analysis_run=False,
                  analysis_source_sha256=protocol.file_hash(Path(__file__)))
    report["timing"] = timing_projection(roster, summary, kind, safety_factor, startup_seconds, chunk_size)
    return report


def load_accepted(protocol_path, root, acceptance_path):
    """Stream, rehash and revalidate all accepted rows; partial acceptance is insufficient."""
    roster = json.loads(Path(protocol_path).read_text())
    protocol.validate_protocol(roster)
    accepted = json.loads(Path(acceptance_path).read_text())
    digest = protocol.file_hash(protocol_path)
    require(accepted["schema"] == ACCEPT_SCHEMA and accepted["kind"] == "all"
            and accepted["complete_roster"] is True and accepted["cross_group_warmup_check"] is True,
            "scientific analysis requires complete all-kind acceptance")
    require(accepted["phase"] == roster["phase"] == "formal", "pilot results cannot enter analysis")
    require(accepted["family_id"] == roster["family_id"] and accepted["protocol_sha256"] == digest
            and accepted["source_sha256"] == roster["source_sha256"], "acceptance identity mismatch")
    files = accepted["files"]
    require(len(files) == accepted["accepted_episodes"] == len(roster["rows"])
            and {record["index"] for record in files} == set(range(len(roster["rows"]))),
            "acceptance does not cover each roster index exactly once")
    records, warmups = [], {}
    for record in sorted(files, key=lambda item: item["index"]):
        path = Path(root) / f"episode_{record['index']:05d}.json"
        require(not path.with_suffix(".error").exists(), "accepted row has software failure artifact")
        require(protocol.file_hash(path) == record["sha256"], "accepted episode SHA mismatch")
        claim_path = path.with_suffix(".claim")
        require(protocol.file_hash(claim_path) == record["claim_sha256"], "accepted claim SHA mismatch")
        result, claim = json.loads(path.read_text()), json.loads(claim_path.read_text())
        validate_episode(result, claim, roster["rows"][record["index"]], roster, digest,
                         accepted["source_commit"])
        update_warmup_checks(warmups, result)
        records.append(compact(result))
    require(sum(record["failed"] for record in records) == accepted["failed_episodes_retained"],
            "accepted failure count mismatch")
    return roster, records, accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True,
                        help="exact namespace directory containing episode_00000.json")
    parser.add_argument("--source", required=True)
    parser.add_argument("--kind", choices=["all", "gpu", "cpu"], default="all")
    parser.add_argument("--safety-factor", type=float, default=2.)
    parser.add_argument("--startup-seconds", type=float, default=300.)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = accept(args.protocol, args.root, args.source, args.kind,
                    safety_factor=args.safety_factor, startup_seconds=args.startup_seconds,
                    chunk_size=args.chunk_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("kind", "accepted_episodes", "failed_episodes_retained")}))


if __name__ == "__main__":
    main()
