# Round16 evaluation implementation addendum — frozen before any evaluation

Implements docs/ROUND16_PROTOCOL.md without changing endpoints, trajectories,
training designs or inference. Files round16_evaluate.py, round16_submit.py,
round16_control_accept.py, round16_queries_accept.py and round16_results.py.

## Roster and identities
Canonical coverage labels dense_short/dense_long/recorded_short/recorded_long/
log_short/log_long/mixed_short/mixed_long plus original(context only).
Order: dataset0..4, these nine labels, Frozen/Updated, seed70/71/72,
wind0/4.9/6.1/8.5,episode0..9. 270models,10,800control rows.
Each original no-LN model is checked against the previously accepted Round13
checkpoint manifest. Short log/mixed models are checked against Round15 full
acceptance; other180 against Round16 full acceptance. All best checkpoints are
ID-selected,39497parameters,strict state load,unchanged norm/split SHA256.
No pilot models enter the canonical roster. Final manifest cannot be constructed
until all180newmodel/360NPZ acceptances and60reusedmodel acceptances are present.
It binds each config/provenance/checkpoint, all source and both protocol hashes.

## Fresh common queries and feedback
40common anchors, wind-major thenepisode0..9. Environment710000+e,
trajectory720000+e,planning730000+e. Physical/noise/trajectory/MPPI functions are
unchanged from accepted Round15/Round11 engines. Warmup100PDsteps;H50,N64;
feedback1500scoredsteps. No wind-specific controller tuning. Each fresh anchor is
independently checked against the feedback engine's PDwarmup state, including
zero wind (gust amplitude uses the original max(wind,.3) convention).

Query noise namespace uses existing function noise_seed_words('a1',160000+i,100,
phase), where phase construct/evaluate remains independent,8samples each.
Original candidate bank and raw predicted trajectories, softmin weights, full
weighted action sequence and physical re-simulation are retained. All270models
receive the same40banks. Numerical failures remain in records; never drop them to
report finite-only factorial means. Acceptance recomputes actions, state/position,
physical and planner costs, solutions, horizon metrics and exact identities.
Feedback acceptance recomputes the unchanged failure-aware capped RMSE and all
length/action/failure/warmup/source checks; all episodes retained.

## Pilot gate and resources
Independent output namespaces; pilots excluded from formal counts.
Query pilot indices0/10/20/30, each full270model roster.
Control pilot dataset0seed70episode0, allnine recipes,two memorymodes,four winds:
72full-length episodes, selected by identities rather than favorable outcomes.
Fullquery pilot must accept4anchors with each <720s; control must accept72episodes
and10timesworst runtime<2880s before submission of ten-episode arraychunks.
These are correctness/runtime gates, not effectiveness thresholds.
Formal query40CPUtasks up to8concurrent; formalcontrol1080RTXtasks×10episodes,
up to20concurrent. Recheck availability/quota before every submission. Other
projects untouched. Hardware changes require separate parity/timing acceptance.
A claim file prevents duplicate execution; software failures must be investigated,
retained and retried in auditable namespaces/releases, not deleted silently.

## Analysis
Complete panel acceptance and per-file hashes are required. Primary analysis
failure-aware tracking RMSE; physical weighted-solution cost, H50queryE and
horizon-averaged physical-position error secondary. Complete roster per metric:
5datasets×9recipes×2memory×3seeds×4winds×10episodes.
Average episode and modelseed within each trainingdataset. All means, all5dataset
scores and descriptive df4t95% nonsimultaneous intervals; no simulationp-values.
Report U−F for everyrecipe; recorded−dense,log−recorded,mixed−log changes in U−F
at eachbudget; long−short changes in U−F perrecipe; budget change in the
mixed−log interaction. Also all within-memory data/budget differences.
Original contrasts are contextual references evaluated on the SAME new tasks;
not inserted as a new factorial level or used to attribute unique causality.
Winds6.1/8.5primary;0/4.9boundary. All adverse/nullcases remain, nooutcome-based
training/evaluation expansion. One-query diagnostics cannot substitute feedback.

## Storage/acceptance execution note — before evaluation
Local disk has6.5GBfree, insufficient for~19GBnewtrainingNPZ plus query/control
traces. Full raw evidence remains in author-owned compute output namespaces. Run the
SAME numerical acceptance functions on scheduled CPUcompute nodes, not login
nodes, and recover complete hash-bound acceptance plus compact checkpoint/model/
configuration/analysis records locally. Use a run-root symlink in the immutable
release's untracked runs/ directory only, so canonical relative hashes remain
identical. This changes data movement and validation location, not any scientific
protocol, acceptance tolerance or inclusion criterion. Downstream traces likewise
must pass full scheduled acceptance before local scientific summaries. Do not
silently delete old local or compute data to make space. Final reproducibility package
must clearly state which raw arrays/traces remain remote and export exact tensors.

Original logged static prediction audit: round16_offline_results.py requires every
accepted NPZ hash and identical flight/center/wind identities across all240matched
models. Report all720checkpoint×wind summaries at3.7/6.1/8.5m/s, exactly28,500windows
perwind; these static-test strata are distinct from zero/4.9feedback boundaries.
The dataset-level recipe/budget contrast helper is shared with feedback/query
analysis to preserve signs and replication units. This is descriptive evaluation
on the unchanged original test set, not additional held-out-data confirmation.
