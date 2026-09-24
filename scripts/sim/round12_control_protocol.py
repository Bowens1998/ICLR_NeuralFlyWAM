"""Immutable Round12B/C rosters, baseline reuse and checkpoint identity contracts."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import round11_closed_loop_protocol as previous  # noqa: E402

from latent_aero_wam.utils.config import config_hash  # noqa: E402

SCHEMA = "round12bc-control-v1"
TRAIN_SOURCE = "275afdc26016a892f58b55edc94d688da5745784"
BASELINE_SOURCE = "87a0333910cef20b4ff2d5918cfdbfe2dba7a4b0"
SETTINGS = {
    "baseline": dict(study="baseline", horizon=50, candidates=64, plant_changes={}),
    "h50_n32": dict(study="B", horizon=50, candidates=32, plant_changes={}),
    "h50_n128": dict(study="B", horizon=50, candidates=128, plant_changes={}),
    "h25_n64": dict(study="B", horizon=25, candidates=64, plant_changes={}),
    "mass_0p9": dict(study="C", horizon=50, candidates=64, plant_changes={"mass": .9}),
    "mass_1p1": dict(study="C", horizon=50, candidates=64, plant_changes={"mass": 1.1}),
    "drag_0p8": dict(study="C", horizon=50, candidates=64, plant_changes={"drag_factor": .8}),
    "drag_1p2": dict(study="C", horizon=50, candidates=64, plant_changes={"drag_factor": 1.2}),
    "tau_0p06": dict(study="C", horizon=50, candidates=64, plant_changes={"tau_omega": .06}),
    "tau_0p10": dict(study="C", horizon=50, candidates=64, plant_changes={"tau_omega": .10}),
    "gust_tau_0p5": dict(study="C", horizon=50, candidates=64, plant_changes={"gust_tau": .5}),
    "gust_tau_2p0": dict(study="C", horizon=50, candidates=64, plant_changes={"gust_tau": 2.}),
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def source_hashes() -> dict:
    """Bind all importable project Python plus the legacy planner and this runner."""
    paths = sorted((REPO / "src").rglob("*.py")) + [
        REPO / "scripts/sim" / name for name in (
            "closed_loop.py", "round11_closed_loop_protocol.py",
            "round12_controls.py", "round12_control_protocol.py")]
    return {str(p.relative_to(REPO)): file_hash(p) for p in paths}


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema") != "round12-checkpoints-v1":
        raise ValueError("unsupported checkpoint manifest schema")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", manifest.get("family_id", "")):
        raise ValueError("family_id must be a simple identifier")
    entries = manifest["checkpoints"]
    if len(entries) != 60:
        raise ValueError("exactly 60 checkpoints are required")
    seeds = manifest["model_seeds"]
    if (len(seeds) != 3 or len(set(seeds)) != 3
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("declare exactly three distinct integer model seeds")
    reuse = manifest.get("reuse_round11_learned", False)
    if reuse and (manifest["family_id"] != "round11_gru" or seeds != [70, 71, 72]):
        raise ValueError("only the original family and seeds may reuse learned baseline")
    expected = {(d, s, update, norm) for d in range(5) for s in seeds
                for update in (False, True) for norm in ("none", "ln")}
    actual = set()
    paths, identities = set(), set()
    for entry in entries:
        key = (entry["dataset_replicate"], entry["model_seed"],
               entry["update_context"], entry["context_norm"])
        actual.add(key)
        path = Path(entry["checkpoint_relpath"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("checkpoint path must remain inside checkpoint root")
        paths.add(str(path))
        identities.add((entry["model_name"], entry["model_seed"]))
        for field in ("checkpoint_sha256", "normalizer_sha256", "config_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", entry.get(field, "")):
                raise ValueError(f"invalid {field}")
        if not re.fullmatch(r"[0-9a-f]{40}", entry.get("source_commit", "")):
            raise ValueError("training source must be an immutable clean commit")
        cfg = entry["config"]
        if (object_hash(cfg) != entry["config_sha256"]
                or config_hash(cfg) != entry["config_hash"]
                or cfg["seed"] != entry["model_seed"]
                or cfg["model"]["name"] != entry["model_name"]
                or cfg["model"]["update_context"] != entry["update_context"]
                or cfg["model"]["context_norm"] != entry["context_norm"]
                or cfg["model"].get("readout_norm") != "none"):
            raise ValueError("checkpoint config does not match manifest identity")
        if not entry.get("split_checksum"):
            raise ValueError("missing training data identity")
        if reuse:
            name = (f"r11b_d{entry['dataset_replicate']}_"
                    f"{'updated' if entry['update_context'] else 'frozen'}_"
                    f"r{int(entry['context_norm'] == 'ln')}")
            if (entry["model_name"] != name or entry["source_commit"] != TRAIN_SOURCE
                    or cfg["model"]["family"] != "context_memory_control"):
                raise ValueError("learned baseline reuse requires exact original model identity")
    if actual != expected or len(paths) != 60 or len(identities) != 60:
        raise ValueError("manifest does not contain the unique full factorial")


def round11_manifest(metadata_root: Path, episode_root: Path, acceptance_path: Path) -> dict:
    """Bind accepted checkpoint hashes without requiring local checkpoint binaries."""
    accepted = json.loads(acceptance_path.read_text())
    if (accepted["accepted_episodes"] != 2560 or accepted["source"] != BASELINE_SOURCE
            or accepted["roster_sha256"] != file_hash(REPO / "configs/sim/round11_closed_loop.json")):
        raise ValueError("Round11D acceptance identity mismatch")
    hashes = {r["index"]: r["sha256"] for r in accepted["files"]}
    if set(hashes) != set(range(2560)):
        raise ValueError("incomplete accepted baseline roster")
    baseline_rows = previous.design()["rows"]
    artifacts = json.loads((REPO / "artifacts/round11b_v1/acceptance.json").read_text())
    entries = []
    for row in baseline_rows[:2400:40]:
        d, seed, name = row["dataset_replicate"], row["model_seed"], row["model"]
        rel = Path(f"round11b_data{d}_v1") / f"{name}_seed{seed}"
        cfg = yaml.safe_load((metadata_root / rel / "resolved_config.yaml").read_text())
        provenance = json.loads((metadata_root / rel / "provenance.json").read_text())
        data = artifacts["replicates"][d]
        expected = dict(git_commit=TRAIN_SOURCE, seed=seed, model_name=name,
                        config_hash=config_hash(cfg), split_checksum=data["split_checksum"],
                        normalizer_sha256=data["normalizer_sha256"])
        if any(provenance.get(k) != v for k, v in expected.items()):
            raise ValueError(f"training provenance mismatch: {name}/{seed}")
        episode_path = episode_root / f"episode_{row['index']:04d}.json"
        if file_hash(episode_path) != hashes[row["index"]]:
            raise ValueError("accepted checkpoint donor episode hash mismatch")
        episode = json.loads(episode_path.read_text())
        if episode["row"] != row or episode["source_commit"] != BASELINE_SOURCE:
            raise ValueError("accepted checkpoint donor identity mismatch")
        entries.append(dict(
            dataset_replicate=d, model_seed=seed, model_name=name,
            update_context=cfg["model"]["update_context"],
            context_norm=cfg["model"]["context_norm"],
            checkpoint_relpath=str(rel / "checkpoint_best.pt"),
            checkpoint_sha256=episode["checkpoint_sha256"], source_commit=TRAIN_SOURCE,
            config=cfg, config_hash=config_hash(cfg), config_sha256=object_hash(cfg),
            split_checksum=data["split_checksum"], normalizer_sha256=data["normalizer_sha256"]))
    manifest = dict(schema="round12-checkpoints-v1", family_id="round11_gru",
                    model_seeds=[70, 71, 72], reuse_round11_learned=True, checkpoints=entries)
    validate_manifest(manifest)
    return manifest


def baseline_contract(acceptance_path: Path) -> dict:
    accepted = json.loads(acceptance_path.read_text())
    if (accepted["accepted_episodes"] != 2560 or accepted["source"] != BASELINE_SOURCE
            or accepted["roster_sha256"] != file_hash(REPO / "configs/sim/round11_closed_loop.json")):
        raise ValueError("invalid baseline acceptance")
    hashes = {r["index"]: r["sha256"] for r in accepted["files"]}
    if set(hashes) != set(range(2560)):
        raise ValueError("incomplete baseline acceptance")
    return dict(source_commit=BASELINE_SOURCE, acceptance_sha256=file_hash(acceptance_path),
                roster_sha256=accepted["roster_sha256"], episode_sha256=hashes)


def design(manifest: dict, baseline: dict, *, phase="formal", pilot_steps=20, studies=None,
           sources: dict | None = None, evaluation_panel=None) -> dict:
    validate_manifest(manifest)
    if phase not in ("formal", "pilot") or not 1 <= pilot_steps <= 1500:
        raise ValueError("invalid phase or pilot length")
    if baseline.get("source_commit") != BASELINE_SOURCE:
        raise ValueError("unrecognized baseline source")
    if set(map(int, baseline["episode_sha256"])) != set(range(2560)):
        raise ValueError("baseline must retain its complete unique roster")
    if evaluation_panel is not None:
        if evaluation_panel != "round13_fresh_v1" or not manifest.get("reuse_round11_learned"):
            raise ValueError("fresh panel requires the original accepted GRU family")
        if studies not in (None, ["baseline"]):
            raise ValueError("fresh panel uses only the unchanged baseline controller")
        rows = []
        for checkpoint_index, entry in enumerate(manifest["checkpoints"]):
            if entry["context_norm"] != "none":
                continue
            for wind in (6.1, 8.5):
                for episode in range(20):
                    rows.append(dict(index=len(rows), setting="baseline", **SETTINGS["baseline"],
                                     kind="model", checkpoint_index=checkpoint_index,
                                     dataset_replicate=entry["dataset_replicate"],
                                     model_seed=entry["model_seed"], model=entry["model_name"],
                                     episode=episode, wind=wind, trajectory_seed=430000+episode,
                                     environment_seed=440000+episode, planning_seed=450000+episode))
        if phase == "pilot":
            rows = [r for r in rows if r["dataset_replicate"] == 0
                    and r["model_seed"] == 70 and r["episode"] == 0]
            rows = [dict(r, formal_index=r["index"], index=i) for i, r in enumerate(rows)]
        return dict(schema=SCHEMA, evaluation_panel=evaluation_panel, phase=phase,
                    family_id=manifest["family_id"], studies=["baseline"], dt=.02,
                    history_steps=100, scored_steps=1500, trajectory_duration=33.02,
                    checkpoint_manifest=manifest, baseline=baseline, baseline_reuse=[], rows=rows,
                    source_sha256=source_hashes() if sources is None else sources,
                    controller_design="fixed_original_SimParams", mppi=previous.design()["mppi"],
                    failure_thresholds=previous.design()["failure_thresholds"], error_cap_m=5.)
    rows = []
    episodes = previous.design()["rows"][-40:]
    learned_reuse = manifest.get("reuse_round11_learned", False)
    if learned_reuse and manifest["family_id"] != "round11_gru":
        raise ValueError("only the accepted Round11 family can reuse learned baseline")
    studies = (["B", "C"] if learned_reuse else ["baseline"]) if studies is None else studies
    if (not studies or len(set(studies)) != len(studies)
            or any(study not in ("baseline", "B", "C") for study in studies)):
        raise ValueError("studies must explicitly select baseline, B and/or C")
    for setting, params in SETTINGS.items():
        if params["study"] not in studies:
            continue
        if setting == "baseline" and learned_reuse:
            continue
        for checkpoint_index, entry in enumerate(manifest["checkpoints"]):
            for ep in episodes:
                rows.append(dict(index=len(rows), setting=setting, **params, kind="model",
                                 checkpoint_index=checkpoint_index,
                                 dataset_replicate=entry["dataset_replicate"],
                                 model_seed=entry["model_seed"], model=entry["model_name"],
                                 **{k: ep[k] for k in ("episode", "wind", "environment_seed",
                                                      "trajectory_seed", "planning_seed")}))
        # PD and preview PD do not depend on the candidate budget or horizon;
        # preview PD executes nominal[0], identical for H=25 and H=50.
        kinds = [] if setting == "baseline" else (
            ["nominal", "true_mean"] if params["study"] == "B" else previous.KINDS[1:])
        for kind in kinds:
            for ep in episodes:
                rows.append(dict(index=len(rows), setting=setting, **params, kind=kind,
                                 **{k: ep[k] for k in ("episode", "wind", "environment_seed",
                                                      "trajectory_seed", "planning_seed")}))
    if phase == "pilot":
        # Timing/contract coverage: each setting, all learned arms, all references,
        # fixed d0/seed70/wind8.5/episode0. Pilot never enters scientific summaries.
        rows = [row for row in rows if row["wind"] == 8.5 and row["episode"] == 0
                and (row["kind"] != "model" or
                     (row["dataset_replicate"] == 0
                      and row["model_seed"] == min(manifest["model_seeds"])))]
        rows = [dict(row, formal_index=row["index"], index=i) for i, row in enumerate(rows)]
    reuse = [dict(index=r["index"], kind=r["kind"], episode=r["episode"], wind=r["wind"],
                  sha256=baseline["episode_sha256"].get(str(r["index"]),
                          baseline["episode_sha256"].get(r["index"])))
             for r in previous.design()["rows"] if learned_reuse or r["kind"] != "model"]
    result = dict(schema=SCHEMA, phase=phase, family_id=manifest["family_id"],
                  studies=studies,
                  dt=.02, history_steps=100, scored_steps=1500 if phase == "formal" else pilot_steps,
                  trajectory_duration=33.02, checkpoint_manifest=manifest, baseline=baseline,
                  baseline_reuse=reuse, rows=rows,
                  source_sha256=source_hashes() if sources is None else sources,
                  controller_design="fixed_original_SimParams",
                  mppi=previous.design()["mppi"],
                  failure_thresholds=previous.design()["failure_thresholds"], error_cap_m=5.)
    return result


def validate_protocol(value: dict, *, verify_sources=True) -> None:
    expected = design(value["checkpoint_manifest"], value["baseline"], phase=value["phase"],
                      pilot_steps=value["scored_steps"], studies=value["studies"],
                      sources=value["source_sha256"], evaluation_panel=value.get("evaluation_panel"))
    if value != expected:
        raise ValueError("protocol differs from the locked complete design")
    if verify_sources and value["source_sha256"] != source_hashes():
        raise ValueError("execution source differs from frozen protocol")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="optional alternate-family checkpoint manifest")
    parser.add_argument("--metadata-root", type=Path, default=REPO / "runs/results")
    parser.add_argument("--round11-episodes", type=Path,
                        default=REPO / "runs/results/round11d_formal_v1")
    parser.add_argument("--round11-acceptance", type=Path,
                        default=REPO / "runs/evidence/round11d_formal_bundle/all_acceptance.json")
    parser.add_argument("--phase", choices=["formal", "pilot"], required=True)
    parser.add_argument("--studies", nargs="+", choices=["baseline", "B", "C"],
                        help="default: B C for original GRU, baseline for alternate family")
    parser.add_argument("--pilot-steps", type=int, default=20)
    parser.add_argument("--evaluation-panel", choices=["round13_fresh_v1"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = (json.loads(args.manifest.read_text()) if args.manifest else
                round11_manifest(args.metadata_root, args.round11_episodes, args.round11_acceptance))
    roster = design(manifest, baseline_contract(args.round11_acceptance),
                    phase=args.phase, pilot_steps=args.pilot_steps, studies=args.studies,
                    evaluation_panel=args.evaluation_panel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(roster, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(phase=args.phase, tasks=len(roster["rows"]),
                          protocol_sha256=file_hash(args.output))))


if __name__ == "__main__":
    main()
