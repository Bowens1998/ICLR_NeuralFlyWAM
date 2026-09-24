# Round18: independent confirmation of the recipe-by-memory-by-budget result

Locked 2026-09-22 before generating new training data or inspecting new outcomes.
The fixed design uses five new independent datasets and 120 new models. This is a
prospective confirmation of a result selected after Round16, not a retrospective
preregistration or a new physical system. Original and confirmation results stay
separate; do not pool them into a post hoc n=10 confirmatory analysis.

## Fixed data and model matrix
Five independently generated datasets, same SimParams, flight durations and
whole-flight train/validation/test design as Round11. Keep four training winds
0/1.3/2.5/4.9 m/s and four trajectories per wind; ID validation uses two separate
trajectories per wind. Retain the original static and changing-wind test families.
Generation trajectory seed = 1810000 + dataset*1000 + group_offset + trajectory,
noise seed = 1820000 + dataset*1000 + group_offset + trajectory; offsets remain
train0/val100/test200. Dataset indices0..4 denote NEW datasets in a new namespace.
Test families share trajectories/noise for paired conditions as previously.
All seeds are distinct from prior generation and feedback integer-seed rosters.

Generate the same 96 training anchors per training flight and eight paired future
noise draws per anchor as Round15. Branch SeedSequence starts with18 instead of15;
remaining components are dataset, original-row index, center and purpose index.
Only Re-simulated (all logged actions) and Mixed (50% candidate actions including
induced future states) are retained. No Dense/Repeated expansion. Full96-anchor
and8-noise mapping, action perturbations, physical simulation and normalizer
fitting are unchanged. New normalizers use only each new training dataset.

2 recipes x2 budgets x2 memory designs x3 seeds(70/71/72) x5 datasets =120models.
No LN, 39,497 parameters, original model/loss/optimizer/dropout/checkpoint rule.
Budgets120/445 logical epochs, 48 batches of256 each epoch =5760/21360 steps;
early stopping disabled, complete OneCycle schedules, original min_delta=.0001.
Pilot: all8 recipe/budget/memory cells for dataset0seed70, separate namespace,
excluded from formal. Both hardware types are probed for FP32 finite forward/
backward and timing; choose one type before formal training on runtime/queue
criteria, never scientific performance. Record all provenance. No precision,
architecture, training-seed or threshold changes based on outcomes.

## Evaluation
All120 checkpoints use exactly40 NEW shared control episodes: winds0/4.9/6.1/8.5,
episodes0..9, environment1830000+e, trajectory1840000+e, planning1850000+e.
Total4800 control episodes. Same100PD warmup/1500 scoring steps, H50,N64,
physical parameters, cost/action constraints and failure-aware capped RMSE.
Retain full original-family ID/static logged prediction for each dataset. Also
run the same common first-decision query panel at these40 anchors for all120
models (4800 model-query records). These query diagnostics are secondary.
Freeze full checkpoint hashes, source hashes and evaluation roster in a separate
addendum before evaluation pilot. No deployment-based model selection.

## Primary analysis and interpretation
Primary endpoint: failure-aware tracking RMSE at6.1/8.5m/s. Within each new dataset,
average three model seeds and ten shared episodes; inference unit is the five
training datasets. Report all five scores and paired contrasts with descriptive,
nonsimultaneous t95 intervals(df4), no p-values. Report all arms and all outcomes.

Primary directional confirmation targets at the long budget, separately at both
primary winds: Mixed minus Re-simulated is negative for Updated and positive for
Frozen. The recipe-by-memory interaction is their difference and expected negative.
Report sign recurrence and interval precision separately. If mean signs recur but
an interval crosses zero, call it directional recurrence with unresolved precision;
do not claim a precise confirmation. If signs fail, weaken the claim explicitly.
Do not require every dataset to have the desired sign, discard a dataset or add
seeds after seeing its result. A precise confirmation of the full opposed-effects
pattern requires all four primary within-design intervals to exclude zero in the
specified directions; this is a descriptive decision rule, not a multiplicity-
adjusted significance test.

Also report U-F within all4 recipe/budget cells, Mixed minus Re-simulated within
each memory design/budget, the corresponding recipe-by-memory interactions,
each arm's long-short change and the budget change in the interaction. Show
short-budget contrasts even if they weaken the story. Winds0/4.9 and query/
logged metrics are secondary; they cannot replace the primary outcomes.
No latent-corruption, universal design rule, hardware closed-loop, independent-
physics or cross-platform claim follows from this confirmation.

## Acceptance and preservation
Complete CSV identity/provenance, train/validation/test disjointness, train-only
normalization, branch physical replay/rotation/target gates before training.
Full finite loss/checkpoint/epoch-step/ID-selection/window-identity and NPZ
numerical acceptance before evaluation. Full query and control trace gates before
analysis. Software failures are documented and corrected, never scored as good
models or silently omitted; finite poor outcomes and physical failures retained.
Keep all raw data/traces on compute and retrieve compact evidence only. Existing
accepted releases/artifacts are immutable.
