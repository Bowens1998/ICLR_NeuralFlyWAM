"""Paired physical-reference contrasts; training datasets remain the inference unit."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from round15_results import interval
from round16_verify_compact import verify


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def analyze(baselines, learned):
    verify(learned)
    assert len(baselines) == 80
    identity = {(r["kind"], r["wind"], r["episode"]) for r in baselines}
    assert identity == {(k, w, e) for k in ["nominal", "true_mean"]
                        for w in [0., 4.9, 6.1, 8.5] for e in range(10)}
    summary = {}
    for wind in [0., 4.9, 6.1, 8.5]:
        block = {}
        for kind in ["nominal", "true_mean"]:
            rr = sorted([r for r in baselines if r["wind"] == wind and r["kind"] == kind],
                        key=lambda x: x["episode"])
            physical = np.asarray([r["rmse_m"] for r in rr])
            assert np.isfinite(physical).all()
            contrasts = {}
            cells = learned["summary"]["rmse_m"][str(wind)]["arm_means"]
            for cell in cells:
                coverage, arm = cell.split("/")
                differences = []
                for d in range(5):
                    per_episode = []
                    for e in range(10):
                        values = [r["rmse_m"] for r in learned["records"] if
                                  (r["wind"], r["coverage"], r["arm"], r["dataset"], r["episode"])
                                  == (wind, coverage, arm, d, e)]
                        assert len(values) == 3
                        per_episode.append(float(np.mean(values))-physical[e])
                    differences.append(np.mean(per_episode))
                contrasts[cell] = interval(differences)
            block[kind] = dict(episode_scores=physical.tolist(), mean=float(physical.mean()),
                               failures=sum(r["failed"] for r in rr),
                               learned_minus_reference=contrasts)
        summary[str(wind)] = block
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--acceptance", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--round16", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    gate = json.loads(a.acceptance.read_text())
    manifest = json.loads(a.manifest.read_text())
    assert not gate["pilot"] and gate["accepted_episodes"] == 80
    assert gate["manifest_sha256"] == sha(a.manifest)
    assert manifest["round16_results_sha256"] == sha(a.round16)
    accepted = {x["index"]: x for x in gate["files"]}
    assert set(accepted) == set(range(80))
    records = []
    for i in range(80):
        p = a.root / f"episode_{i:04d}.json"
        assert sha(p) == accepted[i]["sha256"]
        r = json.loads(p.read_text())
        assert not r["pilot"] and r["row"] == manifest["rows"][i]
        records.append(dict(**r["row"], rmse_m=r["capped_tracking_rmse_m"], failed=r["failed"]))
    result = dict(records=records, summary=analyze(records, json.loads(a.round16.read_text())),
                  acceptance_sha256=sha(a.acceptance), round16_sha256=sha(a.round16),
                  manifest_sha256=sha(a.manifest),
                  caveat="n=5 training datasets, conditional on the shared evaluation roster; physical baselines are not independent training replicates; nonsimultaneous descriptive t95 intervals; no p-values.")
    with a.output.open("x") as f:
        f.write(json.dumps(result, indent=2)+"\n")
    print("accepted and analyzed", len(records))


if __name__ == "__main__":
    main()
