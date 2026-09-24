# Round14 focused follow-up to independent review

Status: architectural training design locked before new-model results;
physical-position/action-source evaluation details pending a separate locked manifest.
This follows a result-informed review, not a preregistration of the original study.

## Question and minimal capacity-controlled architectural comparison
Does continuously accessible original history context change the behavior of an
updating rollout model? Retain original no-LN F/U as historical references.
Train two additional no-LN, width64 GRU decoder arms:

* fixed auxiliary context: c_h = GRUCell([x_h, m0], m_h), m_{h+1}=c_h;
* dynamic auxiliary context: c_h = GRUCell([x_h, m_h], m_h), m_{h+1}=c_h.

Both encode the identical observed history, initialize m_h=m0, use the same head,
state integration, dropout, loss, ID checkpoint selection and training budget.
Both have exactly the same trainable parameter count; all auxiliary inputs have
trainable weights. The dynamic arm exposes no independently retained original
context after h=0. At initialization and identical weights the first-step outputs
must agree. Training is independent and may produce first-step differences.

The additional decoder input matrix has 3*64*64=12,288 weights. Expected total
51,785 vs original39,497. New-arm contrast is the capacity-controlled access
comparison. Comparing a new arm directly with old F/U also changes architecture
and capacity and must not be interpreted as a pure context intervention.
No zero-input dummy weights, width shrinkage, post-hoc capacity adjustment or
outcome-dependent tuning. This is one explicit concatenation implementation,
not universal causal identification of memory pollution or all hybrid models.

## Fixed training roster and selection
Original five Round11B training datasets and three initialization seeds70/71/72:
2 arms x5 datasets x3 seeds=30 new models. Retain train-only normalizers and split
artifacts, original default optimizer/training recipe, ID-only best checkpoint.
Full-length pilot: dataset0, seed70, both arms, same GPU type. Pilot outputs are
separate smoke/pipeline artifacts and excluded from scientific results. Formal
roster must be complete regardless of sign; no significance stopping/extra seeds.
Evaluate original ID and static OOD splits. Scientific focus 6.1/8.5m/s; other
static conditions are reported transparently as secondary if evaluated.

## Evaluation gates and statistics
Before formal training: local unit tests for exact capacity, seed-paired shared
weights, first-step equality, effect of independent context after a memory change,
and gradient flow; immutable source release and source/data/config hashes.
Before release query current B200/RTX/L4 availability, group limits and software
compatibility; select by actual availability. No execution on login nodes.
Pilot requires finite training/validation, valid best checkpoint and full static
NPZ acceptance before formal submission. No overwrite of Round9–13 artifacts.

After all models accepted, lock new evaluation manifests binding checkpoint hashes:
(1) same offline metrics incl physical position, (2) common-query evaluation on
existing primary-wind A1 anchors, (3) original baseline-control trajectories.
Retain controller/objective and noise conventions. Do not start broad LN, plant
perturbation or fresh-episode grids. Compare fixed-aux minus dynamic-aux within
training dataset, average paired seeds/episodes inside each dataset, report all
five differences and descriptive Student-t95 intervals (df4), no window/episode
pseudo-replication and no multiplicity-adjusted discovery claim. Keep offline,
query and control metrics distinct. Publish unfavorable and mixed results.

## Existing-data component analysis (post hoc)
Reproduce original accepted offline aggregates first; then report all components
and terminal h1/5/10/25/50 plus H50 prefix mean at both primary winds. Existing
scalar normalized displacement cannot recover physical meter L2; label honestly.
New inference with true positions is required before calling that gap complete.

## Pending action-source protocol
Preselect anchors without model errors or control outcomes, reconstruct complete
simulator state/history from original trajectory seeds, validate replay first;
compare actual logged future actions with MPPI-bank actions at identical anchors
and paired physical noise. Preserve common noise across models and action sources.
Anchor roster, replay tolerance and full evaluator must be locked before outcomes;
this paragraph does not authorize selecting anchors after inspecting results.
