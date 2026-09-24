# Round14 new context-access model evaluation design

Locked after all30models accepted, before closed-loop or common-query outcomes.
Training acceptance: five round14_access_data*_acceptance.json files (6models
and12NPZeach), source 413b5c1. Predictions may be analyzed now, but do not change
this roster or select models based on OOD scores. Use all ID-selected checkpoints.

## Common-query panel
Use the20accepted original Round12A1 anchors at6.1and8.5m/s (10perwind). Keep exact
saved history, state,64candidateactions and construction/evaluation noise banks.
Score all30newmodels on everyanchor. Include actual softminweightedsolutions with
independent evaluation noise, following Round12A1 procedure; do not approximate
by bestcandidate cost. Original F/U results may be referenced only when action,
noise and anchor identities are exact. New arms have51785parameters; original
arms39497, so the primary matched comparison is fixed-aux vs dynamic-aux.
Full checkpoint/source/protocol/anchortrace hashes must be frozen in executable
manifest, and regenerate original no-LN metrics as a compatibility check before
newmodelvalues are accepted. Physicalcost and predictionmetrics remain separate.

## Closed-loop panel
600newepisodes:30models x2primarywinds(6.1,8.5) x10originalepisodes(0..9).
Use originalenvironment310000+e,trajectory320000+e,planning330000+e;1500scoredsteps,
100stepPDwarmup,H50,N64,identicalMPPIsettings,baselinephysics andfailurethresholds.
Same finite-prediction protection and5metercappedfailure-scoring as original.
No newreferenceepisodes, no newtrajectorydraws, no otherwinds or sensitivitygrid.
OldF/U600correspondingepisodes are reused once as historicalcomparators.
Full-lengthpilot:dataset0seed70,botharms,bothwinds,episode0=fourjobs; separate
pilotnamespace/excludedfromformal. No shortenedpilot forruntimeextrapolation.
Manifest must bindall30accepted bestcheckpoint/config/source/split/normidentities.
Do not modify legacyvalidators to bypass theirexact60modelchecks; use a separate
newmanifest and sourceverification while reusing unchanged run/scoring functions.

## Physical-position offline audit
All originalno-LNF/U and newfixed/dynamicaux models (60total), bothprimarywinds,
all10testflights per matchedtrainingdataset, all2850existingwindows perflight.
Keep parsedhistory/state/actions and originalFP32evaluationcontract. Compute
p_hat=p_current+dt*cumsum(v_hat), compare against recorded physicalp in meters;
also separately report Euler-velocity-integration position reference. This is
position prediction under the paper's integrator, not a newlearnedpositionhead.
Reproduce the accepted normalized errors under a predeclared numericaltolerance
before accepting physicalpositionmetrics. Record inferencehardware/batchsize;
no selection of favorablewindows, no replacement of originalaccepted scores.
The actualloggedfuturemetric in the100anchoractionexperiment is only a small
subset and must not be presented as thisfullaudit.

## Statistics and interpretation
Primarycontrast fixedaux-minus-dynamicaux, lowererror/cost is better. Average
seeds and pairedanchors/episodes within eachtrainingdataset; report allfive
values, mean and descriptive t95(df4), separatewinds. Include absolute metrics,
failures, seedpairs and unfavorableeffects. No pseudoreplication, no p-values or
stopping by significance. Keep parameters-vs-effectivecapacity caveat explicit.
Only assess practical design evidence afterprediction,query andcontrol are all
accepted; no universal mechanism or causal memorypollutionclaim.
