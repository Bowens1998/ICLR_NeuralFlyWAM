# Round17: physical baselines on the Round16 evaluation roster

Locked before any new baseline outcomes, 2026-09-22. This is a result-informed follow-up, not prospective
registration of Round16. Existing training, checkpoints and evaluations are frozen.

## Design
Run Nominal-physics MPPI and True-mean-wind MPPI on the exact Round16 roster:
winds 0/4.9/6.1/8.5 m/s, episodes 0..9, environment seed 710000+episode,
trajectory seed 720000+episode and planning seed 730000+episode. Total 80 episodes.
Use round11_closed_loop.run unchanged: 100 PD warmup steps, 1500 scored steps,
H=50, N=64, unchanged costs/action constraints and failure-aware capped RMSE.
Nominal assumes zero mean wind; True-mean-wind knows only the fixed mean, not gusts
or future noise. It is an informed reference, not an oracle or guaranteed lower bound.
Both physical predictors use existing NumPy implementation, so schedule CPU jobs.
No retraining, parameter tuning or checkpoint selection.

## Gates
Separate pilot: episode 0 at all four winds for both controllers, eight episodes.
Pilot and formal namespaces are separate; pilot is excluded from the 80 results.
Bind source/protocol hashes, the Round16 manifest and accepted compact control
report. Confirm exact row seeds, all raw errors/actions finite where required,
unit quaternions, action limits, failure retention, 1500-step score reconstruction,
warmup state matching Round16, and identical zero-wind outputs for the two
physical baselines. Only a passed full pilot numerical/timing gate permits formal.
Retain failures; no outcome-driven rerun or dropped episode. Raw results remain compute.

## Analysis locked before outcomes
Report all four winds and both references alongside all 18 existing Round16 cells
(8 recipes/budgets x 2 memory designs plus 2 original designs). For each learned
cell and dataset, average the three seed-specific episode scores, subtract each
paired physical-reference episode score, then average the ten episode differences.
Report the five dataset contrasts and descriptive nonsimultaneous t95 intervals
(df=4), conditional on this shared evaluation roster. Do not treat a physical
baseline as five independent fitted models or treat model-seed/episode rows as
independent training replicates. Also report baseline episode scores/means and
failure counts. No p-values or best-cell selection. Primary winds 6.1/8.5; 0/4.9
are boundary checks. Interpret absolute comparisons without rewriting the existing
memory-design contrasts; a better informed physical controller does not refute
the conditional Frozen/Updated finding, and a win does not establish SOTA.
