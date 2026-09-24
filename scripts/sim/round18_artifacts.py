"""Build new whole-flight Round18 manifests after complete generation acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from latent_aero_wam.data.dataset import FlightStore, WindowDataset  # noqa: E402
from latent_aero_wam.data.normalize import Normalizer  # noqa: E402
from latent_aero_wam.data.parser import _read_raw, file_sha256, load_flight  # noqa: E402
from latent_aero_wam.data.splits import manifest_checksum  # noqa: E402
from latent_aero_wam.evaluation.metrics import fit_metric_scales  # noqa: E402
from latent_aero_wam.sim import random_trajectory  # noqa: E402

SOURCE = os.environ.get("LATENT_WAM_SOURCE_COMMIT", "local-development")
GROUPS = {
    "train": "d1_train",
    "val": "d1_val",
    "static": "d2_static_ood",
    "slow": "d3_changing_ood",
    "fast": "d3_changing_ood",
}


def make_manifest(roster, evidence, replicate, root):
    rows = [r for r in roster["rows"] if r["replicate"] == replicate]
    splits = {name: [] for name in ["d0_smoke", *dict.fromkeys(GROUPS.values())]}
    for row in rows:
        splits[GROUPS[row["group"]]].append(
            dict(
                file=row["name"],
                condition=row["condition"],
                trajectory=f"t{row['trajectory_seed']}",
                sha256=evidence[row["index"]]["csv_sha256"],
                start=100,
                stop=2950,
            )
        )
    splits["d0_smoke"] = [{**splits["d1_train"][0], "stop": 356}]
    m = dict(
        data_root=str(root),
        sample_rate_hz=50,
        dt=0.02,
        history_seconds=2.0,
        horizon_seconds=1.0,
        history_steps=100,
        horizon_steps=50,
        decimate=1,
        static_ood_conditions=["30wind", "50wind", "70wind"],
        transfer_rate_mode="backward_causal_v2",
        splits=splits,
        dataset_replicate=replicate,
        generation_source=SOURCE,
        generation_roster=rows,
        generation_roster_sha256=hashlib.sha256(
            json.dumps(roster, sort_keys=True).encode()
        ).hexdigest(),
    )
    m["checksum"] = manifest_checksum(m)
    return m


def accept(root, roster):
    expected = {r["name"] for r in roster["rows"]}
    assert {p.stem for p in root.glob("*.csv")} == expected
    assert {
        p.name.removesuffix(".generation.json") for p in root.glob("*.generation.json")
    } == expected
    evidence = {}
    for row in roster["rows"]:
        path = root / (row["name"] + ".csv")
        e = json.loads((root / (row["name"] + ".generation.json")).read_text())
        assert e["row"] == row and e["params"] == roster["params"]
        assert e["source_commit"] == SOURCE and e["csv_sha256"] == file_sha256(path)
        trajectory = random_trajectory(np.random.default_rng(row["trajectory_seed"]), 60.0, 0.02)
        assert e["trajectory_sha256"] == hashlib.sha256(trajectory.tobytes()).hexdigest()
        np.testing.assert_allclose(_read_raw(path)["p_d"], trajectory, atol=6e-9, rtol=0)
        f = load_flight(path)
        assert f.n_steps == 3000 and abs(f.dt - 0.02) < 1e-8
        for field in ["t", "state", "v_body", "action", "delta_state", "R", "p"]:
            assert np.isfinite(getattr(f, field)).all(), (row["name"], field)
        assert 0 < e["seconds"] < 900
        evidence[row["index"]] = e
    return evidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(f"preserve existing artifacts: {args.output}")
    roster = json.loads((REPO / "configs/sim/round18_generation.json").read_text())
    assert len(SOURCE) == 40
    evidence = accept(args.data_root, roster)  # all370 before writing any artifacts
    args.output.mkdir(parents=True)
    os.environ["LATENT_WAM_DATA_ROOT"] = str(args.data_root)
    records = []
    for replicate in range(5):
        manifest = make_manifest(roster, evidence, replicate, args.data_root)
        split_path = args.output / f"data{replicate}_split.json"
        split_path.write_text(json.dumps(manifest, indent=2) + "\n")
        store = FlightStore(manifest, cache_root=args.output / "cache")
        ds = WindowDataset(manifest, "d1_train", store=store)
        norm = Normalizer.fit(ds.frames())
        norm.metric_scales = fit_metric_scales(ds)
        for stats in norm.stats.values():
            assert np.isfinite(stats.mean).all() and np.isfinite(stats.std).all()
            assert (stats.std > 0).all()
        for scales in norm.metric_scales.values():
            assert np.isfinite(scales).all() and (np.asarray(scales) > 0).all()
        norm_path = args.output / f"data{replicate}_norm.json"
        norm.save(norm_path)
        records.append(
            dict(
                replicate=replicate,
                split_checksum=manifest["checksum"],
                normalizer_sha256=file_sha256(norm_path),
                train_flights=len(ds.blocks),
                train_windows=len(ds),
            )
        )
    (args.output / "acceptance.json").write_text(
        json.dumps(
            dict(source=SOURCE, accepted_flights=len(evidence), replicates=records), indent=2
        )
        + "\n"
    )
    print(json.dumps(records))


if __name__ == "__main__":
    main()
