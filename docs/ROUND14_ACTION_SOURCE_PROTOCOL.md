# Round14 paired action-source evaluation (locked before model outcomes)

Purpose: distinguish changing future actions from changing observation/history
in the original no-LN F/U comparison. This is a conditional action intervention,
not a decomposition of the complete distribution shift in all old experiments.

Roster: every static test flight at6.1/8.5m/s in round11_generation.json:
5dataset families x10flight seeds x2winds=100anchors, exactly step100. No anchors
chosen by error. For each anchor score original F/U, seeds70/71/72, trained on
its matching dataset. H50, FP32, same history and state for both action sources.

Replay original baseline simulator through step150 from original trajectory and
noise seeds. Before any inference, verify CSV SHA against generation.json and
verify p/v/R/w/T_sp/q_sp through step150 against original 8-decimal CSV (absolute
5.1e-9 tolerance). Snapshot atstep100 BEFORE that period's OU innovation.
Model history is parsed originalCSV, aligned exactly with WindowDataset at100;
physical snapshot uses unrounded replay state. Record maximum replay mismatch.
Do not silently relax tolerance; investigate failures and preserve evidence.

Actions: original logged future actions100:150 versus64MPPI candidate sequences
sampled from the original pd_plan nominal atstep100 with an empty warm start.
The planner bank is newly constructed for these logged anchors, not a historical
MPPI bank. Planning seed=510000+roster index. No outcome-dependent changes.
Both sources share exogenous physical noise from SeedSequence([14, index, 0])
with8draws,H50,10substeps; all models share the identical resulting truth bank.
Sources use the same action dtype and quaternion handling. Retain both duplicate
zero candidates as in the original first-decision implementation.

Report scaled velocity/rotation/rate/aggregate, physicalpositionL2meters,
Euler-planner-positionL2meters at h1/5/10/25/50 terminal and prefix means.
Primary source contrast: (U-F) on candidate-source minus (U-F) on logged-source,
separate for6.1/8.5, averaging candidates/noise then pairedseeds/flights within
eachtrainingdataset. Report all5datasetdifferences, descriptive t95(df4), no
p-values, no candidate/window/noise pseudo-replicates. Component/horizon intervals
are descriptive and not simultaneous. Include absolute errors and each source
contrast; an interaction alone does not establish reversal. Failure/nonfinite
results remain explicit; no deleting anchors after observing model results.

Also evaluate the same logged sequence against its actual recorded future p/v/R/w
as a separate subset diagnostic (not the full existing offline test). Fresh-noise
physical branches differ from the realized futurelog. Full offline physical-
position audit requires every existing accepted window and is a separate task.

Before formal evaluation: hash-bound checkpoint/normalizer/source manifest;
replay+batch-alignment+common-noise tests; pilot anchors indices0and1 (two winds
of firstselectedflight) excluded from any timing-based claims. Pilot may be
reused only if all formal settings and artifacts are identical and recorded.
No new training, no adaptive anchor/sample/seed expansion.
