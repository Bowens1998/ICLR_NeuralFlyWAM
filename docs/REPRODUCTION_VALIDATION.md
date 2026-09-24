# Portable-workflow validation

This report distinguishes execution checks from agreement with accepted results.
No model tensors, published numerical records, scientific implementations or
manuscript results were changed by this reproduction-workflow extension.

## Data and model checks

- Both five-dataset cohorts were regenerated from fixed rosters: all 740 CSVs
  and all 160 paired-branch NPZs matched the accepted SHA-256 values exactly.
- All ten training-only normalizers and metric scales were refitted and checked.
  Branch checks cover identity, finite arrays, rotations and delta alignment.
- The original 450 tensor exports remain unchanged. The package verifier checks
  their hashes, strict model loads and the included compact statistics.
- At the first 6.1 m/s query (anchor 20), the new portable entry point reproduced
  all 810 recorded values for 270 bridge models and all 360 values for 120
  confirmation models exactly in CPU FP32.
- The summary adapter reproduced published control statistics from all 2,560
  original, 10,800 bridge and 4,800 confirmation compact records. Missing-episode
  rejection was checked. Offline statistics matched all 240 bridge and 120
  confirmation models. These summary checks use retained records; they are not
  fresh full-roster control runs.
- Concurrent shared-model and manifest publication was checked with eight workers
  and 16 requests, including rejection of conflicting contents.

## GPU execution checks and observed numerical differences

Six dataset-0, seed-70 configurations were selected by design: Frozen and Updated
in the original study, Mixed/long bridge and Mixed/long confirmation. Each ran a
one-epoch training installation check, complete static prediction (85,500
windows), and one full control episode at 6.1 m/s (100 warmup + 1,500 scored
steps). All six control episodes completed without a protocol failure.
Training installation checks are not full-budget retraining and do not establish
that newly trained weights reproduce the paper's full statistics.

These checks used Python 3.11.16, NumPy 2.4.6, PyTorch 2.13.0+cu130 and an NVIDIA
GeForce RTX 3080 Laptop GPU. The accepted original control used L4 GPUs; the
bridge/confirmation control used RTX PRO 6000 GPUs. Scientific source and weights
were checked unchanged. The alternative-machine comparison does not isolate GPU
architecture from other hardware-dependent numerical effects.

The predeclared comparison tolerance was relative 1e-4 and absolute 1e-6. A run
finishing is not counted as passing this numerical comparison. The observed
cross-machine differences are retained below; no tolerance was widened.

| Configuration | Maximum absolute difference in static E | Accepted episode RMSE (m) | Recomputed episode RMSE (m) | Absolute RMSE difference (m) |
| --- | ---: | ---: | ---: | ---: |
| `r11b_d0_frozen_r0` | 1.7415732e-07 | 0.31330916 | 0.27500930 | 0.03829986 |
| `r11b_d0_updated_r0` | 9.4622374e-08 | 0.28600314 | 0.26306309 | 0.02294005 |
| `r16_d0_mixed_long_frozen` | 1.3826415e-05 | 0.23401921 | 0.27632061 | 0.04230139 |
| `r16_d0_mixed_long_updated` | 1.6441569e-05 | 0.10246747 | 0.09584965 | 0.00661782 |
| `r18_d0_mixed_long_frozen` | 9.957701e-06 | 0.19490646 | 0.24962702 | 0.05472055 |
| `r18_d0_mixed_long_updated` | 1.0108575e-05 | 0.12890800 | 0.13312100 | 0.00421301 |

At that tolerance, 10/18 static values and
0/6 control scores matched. This is not a
claim of exact full-trajectory reproduction across hardware. In the original
Frozen example, warmup state differences were below 4e-16 and the first action
differed by less than 5e-13, while later trajectories diverged. The observation
motivates recording the execution environment and retaining all episode results;
it does not change any published scientific result or establish a new mechanism.

For numerical auditing of the reported full rosters, use the supplied accepted
records and compact-statistics checks. For an independent rerun, follow the full
rosters, retain failures, compare dataset-level contrasts and report the execution
environment. A CPU option is available, but matching seeds alone is not a promise
of identical closed-loop trajectories. The software supports the documented
protocol; full-budget retraining and all control episodes were not rerun as part
of this packaging validation.
