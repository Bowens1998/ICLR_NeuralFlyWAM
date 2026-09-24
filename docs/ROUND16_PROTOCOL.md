# Round16: training-recipe bridge — locked before new training

Study design fixed on 2026-09-22.
Result-informed follow-up, not retrospective preregistration of Round11/15.
No original result, dataset split, normalization, model, loss, controller or
conclusion is overwritten. Follow ROUND9_LAUNCH then ROUND16_LAUNCH.

## Question and design
Why do original and Round15 log-only recipes give different mean memory-design
rankings? Separate data support, re-simulation and optimization budget, then test
whether candidate-coverage effects persist at the larger budget.

Four data recipes, crossed with Frozen/Updated no-LN and two budgets:
- dense: draw 12,288 windows without replacement per logical epoch from the full
  original 45,600-window d1_train pool; independently reshuffle each epoch.
- recorded: the Round15 1,536 training anchors, each original recorded window
  repeated eight times, giving 12,288 entries, in Round15 branch order.
- log: exactly the accepted Round15 eight re-simulated logged-action futures per
  anchor, with unchanged histories and actions.
- mixed: exactly Round15 50% candidate exposure, including induced future states.

At every budget, data recipes share 48 full batches of256 per logical epoch,
ID validation cadence and model-seed pairing. Budgets are120/445 logical epochs:
5,760/21,360 optimizer steps. The larger budget equals120 full original training
passes after dropping partial batches (45,600//256=178). This is NOT a literal
reproduction of original early stopping or its validation cadence. Both budgets
use complete OneCycle schedules; budget therefore includes schedule duration and
number of ID-selection opportunities. Do not call it a pure compute-only effect.
Early stopping is disabled by patience=epochs+1. Everything else is unchanged.
Dense vs recorded changes training support/repetition; recorded vs log changes
re-simulated future noise (and numerical replay); log vs mixed changes future
actions and their resulting states. These conditional contrasts are not a fully
crossed density×noise experiment, nor a unique mechanistic decomposition.

Five existing independent training datasets, seeds70/71/72, all39,497parameters.
Full matrix240models. Reuse the60 accepted Round15 short-budget log/mixed models
without retraining; train180 new models (36/dataset). Reused models retain original
source/hardware provenance; verify the new loader's log/mixed mapping is unchanged.
Pilot16models =dataset0seed70×four data recipes×two budgets×two memory modes,
separate output namespace, excluded from formal. No outcomes used to tune design.
All formal new training uses RTX Pro6000, as did accepted Round15. B200 remains an
alternative only after fresh compatibility/timing acceptance and balanced hardware
assignment; no architecture is assigned selectively by treatment.

## Acceptance before downstream evaluation
Hash-bound immutable source/configs, original split/normalizer checksums, accepted
coverage NPZ hashes. Validate recorded targets exactly equal original dataset
items, dense sampling stays within training and has no repeated index per epoch,
log/mixed item mapping unchanged. All train losses, validation, parameters finite;
exact epoch/step/batch counts; best checkpoint reconstructed solely from original
ID min_delta=.0001. Preserve bad finite models and failures; software failures are
fixed and recorded, not filtered by performance. Full original ID/static offline
NPZ identities and numerical summaries required before formal downstream use.

## Evaluation and prespecified analysis
Fresh seed base environment710000+e, trajectory720000+e, planning730000+e,
e=0..9. Winds0/4.9/6.1/8.5m/s. Original30 no-LN checkpoints plus all240 bridge
checkpoints evaluated on the SAME 40 fresh feedback episodes (10,800 total).
Original models are contextual anchors, not factorial cells. Existing30s scoring,
100PDwarmup,N64,H50,failure-aware cap and all controller settings unchanged.
Also evaluate fresh common first-decision candidate banks with independent physical
noise, actual weighted returned solutions, query E and physical position. Retain
original offline tests; no deployment-based selection. An implementation addendum
must freeze checkpoint hashes, complete row roster, source and evaluation seeds
before evaluation pilots; no outcome-driven expansion.

Primary endpoint: failure-aware tracking RMSE, winds6.1/8.5. For each budget,
report U−F within every recipe, and changes in U−F for recorded−dense, log−recorded,
mixed−log. Report each recipe's long−short change in U−F and the budget change
in the mixed−log interaction. Always report all arm means and absolute within-
architecture data/budget changes; do not substitute interaction for absolute gain.
Query cost/E/position are secondary diagnostics, not feedback surrogates.
Winds0/4.9 are boundary checks. Average model seeds and episodes WITHIN each of
five training datasets; report all five paired scores, descriptive nonsimultaneous
t95% intervals(df4), no simulation p-values, no new seed selection. Keep all cases
including reversal of the desired story. This study cannot identify latent-memory
corruption, establish a universal freezing rule or eliminate single-physics scope.

## Next-stage limits
Complete and analyze this bridge before choosing any additional training question.
No automatic broad benchmark, hidden-size sweep or new physical mechanism claim.
If the bridge supports a more precise design boundary, integrate it within9 main
pages; otherwise report the uncertainty and retain the current bounded conclusions.
