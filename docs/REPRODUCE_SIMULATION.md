# Reproduce the simulation studies with independent compute

No dataset download, personal repository, access request or cluster account is
needed. The supplied simulator generates the CSVs and paired branch data from
fixed seed rosters. Large generated data and full traces are deliberately not
distributed. The scripts below use the existing model, optimizer, simulator,
controller, scoring and statistical-analysis implementations without changing
their scientific definitions.

Run from the repository root after the README installation and `verify_core.py`.
Use the pinned dependencies for data regeneration: each generated CSV must match
the accepted SHA-256 before proceeding. The pipeline stops on a mismatch; do not
replace expected hashes to make it pass. Cross-platform floating-point behavior
can require reproducing the pinned environment. Training on different hardware
is not guaranteed to reproduce identical weights or scores.
Closed-loop scores can also change when evaluating the same weights on another
GPU. Tiny action differences feed into subsequent states and planner queries;
matching seeds does not guarantee matching individual trajectories. Evaluation
outputs record the GPU, library versions and floating-point backend settings.
The validation report distinguishes numerical agreement from a completed run.

## Scope and output isolation

| Study | Cohort | Model runs | Complete control roster |
| --- | --- | ---: | ---: |
| `original` | original five datasets | 60 (including normalization controls) | 2,560, including 160 physical references |
| `bridge` | same original five datasets | 270 (240 factorial models plus 30 original references) | 10,800 |
| `confirmation` | five new datasets | 120 | 4,800 |

The bridge includes the original no-normalization models and reuses short-budget
models. The unique exports remain 450 across all packaged extensions. Do not
retrain an overlapping model twice in one workspace. Use a separate workspace
per study when retraining a full roster. Query reproduction here covers the 40
fresh first-decision anchors for bridge and confirmation. Other diagnostics,
real-flight-log experiments and non-GRU extensions retain their historical
scripts and evidence; this is not an end-to-end entry point for every appendix.
The bridge offline factorial has 240 models; its 30 original references are
evaluated in the original offline panel. Use `plan --study bridge --stage offline`
or the roster driver's `--dry-run` to inspect stage-specific model indices.

`reproduction/` is the default output directory. Set `--workspace /path/to/work`
before the subcommand to use another disk. Nothing writes to packaged evidence.
`init` creates runtime split copies with valid checksums after anonymization and
copies the accepted training-only normalizers unchanged. The historical split
identity is recorded separately. Model adaptation adds local configuration and
provenance to the tensor-only exports; it does not alter model tensors or invent
optimizer states. Original and newly computed results stay separate.

## 1. Generate and verify data

For original and bridge, run:

```bash
python scripts/repro/run_study.py init --cohort original
python scripts/repro/run_study.py data --cohort original
python scripts/repro/run_study.py branches --cohort original
python scripts/repro/run_study.py validate-data --cohort original --with-branches
```

For confirmation, repeat these four commands with `--cohort confirmation`.
The original study itself does not use branch training; its `branches` command
and `--with-branches` flag can be omitted.

Each cohort has 370 CSV tasks (indices 0–369) and 80 branch tasks (indices 0–79).
For example, `data --cohort original --indices 0 1 2` runs three CSV tasks. Shard
disjoint indices across CPU workers, then wait for all CSV tasks before branch
generation, and all branch tasks before `validate-data`. Each dataset has 74 CSV
tasks and 16 branch tasks. `validate-data --datasets 0 --with-branches` validates
one complete dataset for an initial run. Validation checks raw hashes, refits
the training-only normalizer and metric scales, and checks branch identities,
shapes, finite values, rotations and delta alignment. It retains every branch,
including threshold exceedances. Data-generation restarts verify existing
completed files; partial files require inspection and a clean task directory.

The original CSV corpus occupies about 564 MiB and its branch files about
778 MiB. Allow several GiB per cohort including cache. Offline traces and full
training/evaluation outputs require additional storage; plan this on your own
compute system. CPU generation and one GPU per model/episode worker suffice;
distributed training is not required.

## 2. Use accepted weights or retrain

The default evaluation uses `--weights pretrained` and all supplied weights.
Retraining is optional for checking the reported evaluation; it is required for
an independent training replication.

For a single full training run:

```bash
python scripts/repro/run_study.py train --model r11b_d0_frozen_r0 --seed 70
python scripts/repro/run_study.py train --model r16_d0_mixed_long_updated --seed 70
```

The wrapper resolves each exact training configuration from
`CHECKPOINT_EXPORTS.json`: original stopping/selection rules, or complete
120/445-epoch bridge schedules, batch size 256, gradient accumulation 1 and FP32.
The fixed model seeds are 70, 71 and 72. `--device cpu` is available, but full
training and control are substantially slower. GPU requests fail explicitly if
CUDA is unavailable. `--workers` changes data-loading parallelism only.

Inspect the full plan or distribute independent model indices:

```bash
python scripts/repro/run_study.py plan --study bridge
python scripts/repro/run_roster.py --study bridge --stage train --dry-run
python scripts/repro/run_roster.py --study bridge --stage train --indices 0 1 2
```

Omit `--indices` to execute the entire roster sequentially, or submit disjoint
indices to your scheduler. Train each study in its own workspace if running all
three. Existing training/result directories are protected against overwrite.
The driver stops on failure; it does not silently omit or replace failed runs.

## 3. Evaluate prediction, queries and control

Examples with supplied weights:

```bash
python scripts/repro/run_study.py offline --model r11b_d0_frozen_r0 --seed 70
python scripts/repro/run_study.py control --study original --index 0
python scripts/repro/run_study.py query --study bridge --index 20 --device cpu
```

Offline evaluation uses every static-OOD window at stride 1 (85,500 windows per
model, 28,500 per wind). `--split d1_val` and `--split d3_changing_ood` are also
available. Control uses the fixed published episode row, 100-step shared PD
warmup, horizon 50, 64 candidates and 1,500 scored steps. It keeps failures in the
capped tracking score. Query generation uses the original paired noise draws
and independent construction/evaluation draws. `--save-traces` optionally saves
offline/query arrays; otherwise numerical summaries are retained.

For the complete bridge evaluation and paired summaries:

```bash
python scripts/repro/run_roster.py --study bridge --stage offline
python scripts/repro/run_roster.py --study bridge --stage query --device cpu
python scripts/repro/run_roster.py --study bridge --stage control
python scripts/repro/run_study.py summarize --study bridge --panel offline
python scripts/repro/run_study.py summarize --study bridge --panel query
python scripts/repro/run_study.py summarize --study bridge --panel control
```

Replace `bridge` by `confirmation` for the prospective cohort. For `original`,
run offline and control (the general query command is not offered). Add
`--weights retrained` to evaluation and summarization commands to use newly
trained checkpoints. Retrained configurations are checked against the complete
protocol; smoke checkpoints cannot be used as formal checkpoints.

Summarization requires the complete roster and computes seed/episode averages
within each of five training datasets before the paired contrasts. It reuses
the published descriptive t-interval calculations and does not treat windows
or episodes as independent training replicates. Compare new summaries with the
packaged `ROUND11*`, `ROUND16*` and `ROUND18*` records; newly computed results are
never written over those records. Full training is not expected to produce
bitwise-identical results across GPU/library versions.

## 4. Small installation checks

After generating and validating dataset 0, run:

```bash
python scripts/repro/run_study.py train --model r11b_d0_frozen_r0 --seed 70 --smoke
python scripts/repro/run_study.py offline --model r11b_d0_frozen_r0 --seed 70 --smoke
python scripts/repro/run_study.py control --study original --index 0 --smoke
```

Smoke training uses one epoch, 512 sampled training examples and sparse
validation. Smoke prediction evaluates 256 windows; smoke control scores only
three steps after the unchanged warmup. These are execution checks, not paper
results. They are marked and stored in `smoke_runs/` or `smoke_results/`, which
formal summary commands cannot read. Full-budget reruns remain the reviewer's
responsibility when independently reproducing training.

## Anonymity

All repository-internal links are relative. This workflow contains no personal
repository, release-asset, cloud-storage or authenticated data download URL.
The anonymous repository entry is sufficient for retrieving the source and
weights. Runtime paths and device details describe the machine performing the
reproduction; do not publish generated runtime metadata without reviewing it.
