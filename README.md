# When Memory Updates Help Prediction but Hurt Control: A Flight Dynamics Study

Anonymous reproduction code and accepted numerical records for the submitted
manuscript. The comparison follows Frozen and Updated memory designs through
logged-action prediction, planner queries, and closed-loop flight control.

## Quick start

Use Python 3.11. From this directory, create an environment and install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-replay.txt
python -m pip install --no-deps -e .
python verify_core.py
python scripts/repro/replay_confirmation_query.py
```

The requirements file records the dependency versions used for the local
verification. Install a PyTorch build suitable for your operating system before
the remaining dependencies if needed. The replay uses CPU FP32 and does not
require a GPU. Run the commands from the repository root.

`verify_core.py` checks every packaged file against `PACKAGE_MANIFEST.json`,
strictly loads all 450 model exports, and reconstructs the included compact
statistics. Run it before regenerating reports or figures: analysis commands can
overwrite packaged reports, in which case their original manifest hashes will
no longer match. Start from a fresh extraction when repeating integrity checks.

The query replay reconstructs all 120 confirmation models at one complete
first-decision anchor, using only the supplied tensors and simulator code. It
compares 360 prediction, position, and physical returned-solution-cost values
with accepted records. The first episode at 6.1 m/s (anchor 20) was selected
before replay. The fixed tolerances are relative `1e-4` and absolute `1e-6`.
The tested environment reproduced all 360 values exactly. The output records
actual discrepancies and dependency versions; bitwise identity is not promised
across environments. The output is created without overwriting an existing file;
use `--output another_result.json` for an additional run.

## What is included

| Component | Contents |
| --- | --- |
| `src/latent_aero_wam/` | Models, normalization, datasets, training, evaluation, and flight simulator |
| `checkpoints/` | 450 unique tensor-only GRU model exports |
| `CHECKPOINT_EXPORTS.json` | Model identities, configurations, normalizers, and original/export hashes |
| `configs/`, `artifacts/` | Resolved study configurations, data splits, normalizers, and evaluation rosters |
| `reports/` | Accepted numerical records and complete contrast summaries |
| `runs/evidence/` | Portable copies of numerical acceptance records |
| `docs/` | Protocols specifying interventions, rosters, acceptance, and statistical analysis |
| `scripts/analysis/` | Analysis and independent compact-record verification |
| `scripts/sim/` | Data generation, physical queries, and control evaluation |
| `scripts/paper/`, `paper/figures/` | Table and editable vector-figure generators |
| `scripts/repro/` | Self-contained confirmation-query replay |

The 450 exports comprise 60 original, 30 auxiliary-context, 60 action-coverage,
180 additional training-bridge, and 120 independent-confirmation models. The
training-bridge roster reuses some earlier models; each distinct checkpoint is
exported only once. The original and confirmation cohorts each contain five
independently generated training datasets and are analyzed separately.

Study identifiers in filenames connect the paper to the supporting evidence:

| Paper content | Supporting records and entry points |
| --- | --- |
| Original reversal and normalization controls | `ROUND11*`, `ROUND13_EXISTING_RESULTS.json`, `scripts/analysis/round13_existing.py` |
| Candidate queries, later-query strata, and robustness | `ROUND12*`, `ROUND13_QUERY_BOUNDARIES.json`, `ROUND13_FRESH_RESULTS.json` |
| Physical position, action source, and auxiliary context | `ROUND14*` |
| Initial matched action-coverage experiment | `ROUND15*` |
| Four-recipe, two-budget training bridge; main Table 1 | `ROUND16*`, `scripts/analysis/round16_verify_compact.py` |
| Matched physical-controller references | `ROUND17*`, `scripts/analysis/round17_verify_compact.py` |
| Prospective five-dataset confirmation; main Table 2 | `ROUND18*`, `scripts/analysis/round18_verify_compact.py` |

## Figures and tables

These commands regenerate the four included figures as PDF and editable SVG:

```bash
python paper/figures/figure1_architecture.py
python scripts/paper/round16_evidence_figure.py
python scripts/analysis/round11d_figure.py
python scripts/paper/round18_confirmation_figure.py
```

For the confirmation tables, run:

```bash
python scripts/paper/round18_tables.py
```

Other table generators are in `scripts/paper/`. The separate Overleaf archive
contains the complete manuscript sources, bibliography, figure PDFs, and editable
SVGs. This code repository does not contain the full manuscript.

## Reproduction scope and provenance

The package supports exact model loading, reconstruction of published compact
statistics, regeneration of figures and tables, and the executable single-anchor
query replay described above. It does not include raw flight logs, generated
training CSVs, branch/query NPZ collections, full control traces, real-log model
checkpoints, or non-GRU model checkpoints. The other training and raw-evaluation
scripts are provided as the executed methodology; they require those external
inputs and are not complete end-to-end runs from this archive alone. Public
flight logs originate from O'Connell et al., *Science Robotics* (2022), Neural-Fly.

Model tensors and all experimental numbers are unchanged. Tensor-only exports
omit optimizer states, random-generator states, and private path metadata.
Their hashes therefore differ from the original full checkpoints. Original
checkpoint and raw-evidence SHA-256 values are retained as provenance references;
only `PACKAGE_MANIFEST.json` specifies hashes of the actual packaged files.

Published paths use neutral names. Historical repository commit identifiers are
consistently replaced by opaque 40-character provenance tokens, including in
the legacy validation scripts. These tokens preserve identity comparisons but
are not Git commit addresses. Scheduler identifiers are anonymized. These
portability changes mean historical raw-file hashes do not hash the transformed
copies. No Git history, credentials, account information, or institution-specific
deployment configuration is distributed.

## AI Use Disclosure

OpenAI tools were used for language editing and to assist with implementing portions of analysis and visualization code specified by the authors. All outputs were reviewed and validated by the authors.

## Reproducibility statement

The following statement is repeated from the manuscript, with its equation and
appendix references resolved:

Equations 1–3 define the controlled model and outcome. Appendices A–H specify
preprocessing, training, statistical comparisons and simulation. Versioned
manifests, training-only normalizers, per-run source hashes, evaluation artifacts
and analysis scripts record the executed studies. All 24 planned fixed-log
contrasts are included; simulation summaries retain all arms and the complete
episode roster. The accompanying anonymous core supplement contains all 60
original, 30 auxiliary-context, 60 action-coverage, 180 training-bridge and 120
independent-confirmation GRU model tensor exports (450 unique models), resolved
configurations, real-log and simulation split manifests, normalizers, accepted
numerical records and scripts for the paired analyses and figures. Its README
and hash manifest specify exact reproduction commands and distinguish portable
exports from original accepted files. A self-contained replay reconstructs all
120 model evaluations at one first-decision query from exported tensors and
simulator code, reproducing 360 prediction, position and physical solution-cost
values exactly in the tested environment. Raw flight logs, full simulation/query
traces and non-GRU checkpoints are outside this compact package; it supports
auditing the core comparison, not complete end-to-end reproduction of every
extension from the package alone.
