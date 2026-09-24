"""Physical MPPI references matched exactly to the accepted Round16 roster."""
# ruff: noqa: E402
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/sim"))
import round16_evaluate as old

sha = old.sha
WINDS = old.WINDS


def rows():
    result = []
    for kind in ["nominal", "true_mean"]:
        for wind in WINDS:
            for episode in range(10):
                result.append(dict(index=len(result), kind=kind, wind=wind, episode=episode,
                                   environment_seed=710000+episode, trajectory_seed=720000+episode,
                                   planning_seed=730000+episode))
    return result


def sources():
    names = ["scripts/sim/round17_evaluate.py", "scripts/analysis/round17_accept.py",
             "scripts/analysis/round17_results.py", "scripts/evidence/round17_submit.py",
             "docs/ROUND17_PROTOCOL.md"]
    return {**old.sources(), **{n: sha(ROOT / n) for n in names}}


def design():
    previous = json.loads((ROOT / "configs/sim/round16_evaluation.json").read_text())
    keys = ["wind", "episode", "environment_seed", "trajectory_seed", "planning_seed"]
    assert {tuple(r[k] for k in keys) for r in rows()} == {
        tuple(r[k] for k in keys) for r in previous["rows"]}
    return dict(schema="round17-baselines-v1", rows=rows(), source_sha256=sources(),
                control_pilot=[r["index"] for r in rows() if r["episode"] == 0],
                round16_manifest_sha256=sha(ROOT / "configs/sim/round16_evaluation.json"),
                round16_results_sha256=sha(ROOT / "reports/ROUND16_CONTROL_RESULTS.json"))


def validate(m):
    assert m == design(), "Manifest/source or reference mismatch"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--index", type=int)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    if a.write_manifest:
        with a.manifest.open("x") as f:
            f.write(json.dumps(design(), indent=2)+"\n")
        return
    m = json.loads(a.manifest.read_text())
    validate(m)
    assert a.index in range(80)
    if a.pilot:
        assert a.index in m["control_pilot"]
    source = os.environ["LATENT_WAM_SOURCE_COMMIT"]
    assert len(source) == 40 and all(c in "0123456789abcdef" for c in source)
    a.output.mkdir(parents=True, exist_ok=True)
    stem = a.output / f"episode_{a.index:04d}"
    identity = dict(index=a.index, manifest_sha256=sha(a.manifest),
                    execution_source=source, pilot=a.pilot, panel="control")
    with stem.with_suffix(".claim").open("x") as f:
        f.write(json.dumps(identity))
    try:
        torch.set_num_threads(1)
        r = old.diag.original_runner.run(m["rows"][a.index], device="cpu", scored_steps=1500)
        r.update(identity, schema=m["schema"], source_sha256=m["source_sha256"],
                 checkpoint_sha256=None)
        with stem.with_suffix(".json").open("x") as f:
            f.write(json.dumps(r, indent=2, allow_nan=False)+"\n")
        print(a.index, r["failed"], r["elapsed_seconds"])
    except Exception:
        stem.with_suffix(".error").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
