# Round15: training action coverage × rollout memory

Designed after Round14 and narrative reorganization. Result-informed prospective
follow-up; not preregistration of the entire campaign. Freeze before formal training.

## Question and contrast
Does covering planner-like future actions during training change the Updated-minus-
Frozen query/control gap? Four no-LN original GRU arms (39,497parameters each):
Frozen/Updated × log-only/mixed-action training. All60models =5original training
datasets ×3seeds70/71/72 ×4arms; train from scratch, never modify old weights.
Primary interaction at each wind: (Updated−Frozen)_mixed −(Updated−Frozen)_log.
Report all four absolute means and paired dataset scores. Negative interaction
indicates relative improvement for updating; absence of a detectable interaction
is not proof of equivalence or an identified latent mechanism.

## Matched branch dataset
Only original d1_train flights and winds0/1.3/2.5/4.9m/s. For each of16flights per
dataset, choose96equally spaced integer centers from its valid original training
block, including endpoints. No test/validation histories, states, trajectories or
wind values may generate training branches. Replay original simulator/gust state
and validate every recorded p,v,R,w,throttle,quaternion field to5.1e-9 (8-decimal
CSV precision); preserve replay failures. Histories/current model inputs come
from the original parser. No normalization or split file is refit or overwritten.

At each center generate a fresh64-candidate bank with the unchanged MPPI proposal,
PD nominal plan, empty warm start, H50 and sigma(.04,.08,.08). Uniformly choose8
candidate indices without replacement using outcome-independent seeds. Generate8
fresh physical-noise draws. Branchr pairs the logged action sequence and sampled
candidate r under identical noise drawr/full initial state/gust. Float32actions
are used by both simulator and model; physical truth uses10substeps.
Seeds: proposal[15,dataset,original_row_index,center,0], candidate selection same
with suffix1, physical noise same with suffix2. Original CSV noise/trajectories
are separate. Store both outcomes, physical positions and action identities.
All finite branches retained including threshold exceedances; no outcome filtering.

Log-only selects all8logged branches. Mixed selects logged for branch0..3 and
candidate for branch4..7:50%candidate exposure, equal12,288windows per dataset.
Both regimes have identical history multiplicities, target counts, exogenous
noise and sample ordering. It is a controlled branch-data comparison, NOT a
literal rerun of the original45,600overlapping logged training windows. New
log-only training is necessary to control re-simulation, sample count and budget.
Training normalization remains the original training-only normalization in both
arms; the treatment includes future state changes induced by different actions.
It cannot isolate action coordinates while holding their physical consequences fixed.

## Optimization and acceptance
Same original losses,AdamW,lr.001,weightdecay.0001,clip1,OneCycle,FP32,batch256.
Exactly120epochs/48batches per epoch=5760optimizer steps per model (gradaccum1);
early-stopping patience121 prevents unequal stopping. ID checkpoint selection
uses unchanged original d1_val windows/stride5/aggregate/min_delta.0001, never
candidate/OOD/control selection. Same random initialization and shuffled index
stream within each seed; check all trainable tensor dimensions and active grads.
Training both regimes on the same GPU architecture per formal block; pilots on
RTX/B200 may benchmark compatibility before choosing one for all formal arms.
No precision relaxation or hyperparameter tuning based on the main outcomes.

Data pilot must validate replay, shapes, source/config/hash/seed/center identities,
SO(3), finite tensors and delta reconstruction. Scalar physical replay against
the vectorized branch integrator verifies one full H50branch including noise.
Train pilot uses all4arms dataset0seed70, full120epochs, excluded from formal.
Accept finite loss/val/checkpoints and exact5760steps; bad finite performance is
retained, not a reason to change recipes. Preserve failures; fix software causes
in immutable new releases, never overwrite accepted outputs.

## Evaluation (locked before observing trained outcomes)
Primary windy strata6.1/8.5m/s; secondary4.9m/s. Common-query panel:10fresh first-
decision anchors per wind, seeds environment610000+e,trajectory620000+e,
planning630000+e, e=0..9, H50/N64. Every checkpoint receives the same history and
bank. Eight construction and eight independent evaluation noise draws for
physical weighted-solution cost and query E/physical-position error. Record all
component/horizon outcomes and failures; original trained no-LN30models may be
scored as historical contextual references, not included in the primary factorial.

Closed-loop:60models ×3winds ×10same fresh episodes=1800learned episodes; original
30second scoring after100PDwarmup, unchanged MPPI budgets and failure cap.
Seeds above; no checkpoint/controller tuning. Offline evaluation includes original
ID and complete static test splits; coverage effects need not improve log errors.
Primary summaries: query E, actual weighted-solution physical cost and tracking
RMSE; paired interaction per wind. No p-values or outcome-based seed expansion.
Average seeds and anchors/episodes within each training dataset, then descriptive
Student-t95% CI(df4) across5dataset differences. Do not treat windows/branches as
independent replications; report non-simultaneous intervals and all signs.

## Interpretation and completion
If mixed coverage shrinks the gap, revise the claim toward distribution matching;
if it does not, limit the statement to this50%exposure recipe, one physics family
and tested proposals. No support either way for universal freezing, an identified
memory-corruption mechanism or universal OOD robustness. This test does not cover
policy-induced history shifts or later warm-start distributions during training.
After all accepted outcomes, integrate only what bears on the core claim, compile
within9main pages, refresh editable packages and perform a final local review.
