"""Full paired-branch shape/identity/rotation/delta acceptance, no score filtering."""
# ruff: noqa: E402 -- standalone entry point resolves repository imports first.

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from latent_aero_wam.utils.rotation import matrix_to_rot6d_np, so3_log_np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def accept(root, design, source, indices):
    records = []
    for index in indices:
        stem = root / f"flight_{index:03d}"
        j = json.loads(stem.with_suffix(".json").read_text())
        spec = design["rows"][index]
        assert j["status"] == "complete" and j["spec"] == spec and j["source"] == source
        assert (
            j["protocol_sha256"] == design["protocol_sha256"]
            and max(j["replay_errors"].values()) <= 5.1e-9
        )
        path = stem.with_suffix(".npz")
        assert sha(path) == j["trace_sha256"]
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(z["centers"], spec["centers"]) and z[
                "candidate_indices"
            ].shape == (96, 8)
            for row in z["candidate_indices"]:
                assert len(set(row.tolist())) == 8 and row.min() >= 0 and row.max() < 64
            for key in z.files:
                assert np.isfinite(z[key]).all(), key
            for kind in ["logged", "candidate"]:
                state, R, d = z[kind + "/state"], z[kind + "/R"], z[kind + "/delta"]
                assert (
                    state.shape == (96, 8, 50, 12)
                    and R.shape == (96, 8, 50, 3, 3)
                    and d.shape == (96, 8, 50, 9)
                )
                assert z[kind + "/p"].shape == (96, 8, 50, 3) and z[kind + "/threshold"].shape == (
                    96,
                    8,
                )
                assert np.max(np.abs(np.swapaxes(R, -2, -1) @ R - np.eye(3))) < 1e-6
                assert np.max(np.abs(np.linalg.det(R) - 1)) < 1e-6
                np.testing.assert_allclose(
                    state[..., 3:9], matrix_to_rot6d_np(R), atol=1e-7, rtol=1e-6
                )

                def previous(x, k):
                    return np.concatenate(
                        [
                            np.broadcast_to(
                                z["current_" + k][:, None, None], (96, 8, 1) + x.shape[3:]
                            ),
                            x[:, :, :-1],
                        ],
                        axis=2,
                    )

                dv = state[..., :3] - previous(state[..., :3], "v")
                dw = state[..., 9:] - previous(state[..., 9:], "w")
                dr = so3_log_np(np.swapaxes(previous(R, "R"), -2, -1) @ R)
                np.testing.assert_allclose(
                    d, np.concatenate([dv, dr, dw], axis=-1), atol=3e-6, rtol=2e-4
                )
            assert z["logged/actions"].shape == (96, 50, 5) and z["candidate/actions"].shape == (
                96,
                8,
                50,
                5,
            )
            record = dict(
                index=index,
                flight=spec["block"]["file"],
                dataset=spec["dataset"],
                npz=path.name,
                sha256=j["trace_sha256"],
                threshold_exceedances={
                    k: int(z[k + "/threshold"].sum()) for k in ["logged", "candidate"]
                },
            )
        records.append(record)
    return dict(
        accepted_files=len(records),
        accepted_windows_per_regime=len(records) * 768,
        source=source,
        files=records,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--design", type=Path, default=ROOT / "configs/sim/round15_branch_design.json")
    ap.add_argument("--source", required=True)
    ap.add_argument("--indices", nargs="+", type=int)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--manifests", type=Path)
    args = ap.parse_args()
    m = json.loads(args.design.read_text())
    report = accept(args.root, m, args.source, args.indices or list(range(80)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.manifests:
        assert report["accepted_files"] == 80
        args.manifests.mkdir(parents=True, exist_ok=True)
        for d in range(5):
            split = ROOT / f"artifacts/round11b_v1/data{d}_split.json"
            j = dict(
                schema="round15-accepted-training-v1",
                dataset=d,
                data_root=str(args.root.resolve()),
                original_split_checksum=json.loads(split.read_text())["checksum"],
                original_split_sha256=sha(split),
                normalizer_sha256=sha(ROOT / f"artifacts/round11b_v1/data{d}_norm.json"),
                design_sha256=sha(args.design),
                protocol_sha256=m["protocol_sha256"],
                files=[f for f in report["files"] if f["dataset"] == d],
            )
            with (args.manifests / f"data{d}_coverage.json").open("x") as f:
                f.write(json.dumps(j, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}))


if __name__ == "__main__":
    main()
