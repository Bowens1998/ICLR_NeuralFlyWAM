# Round18 evaluation implementation addendum

Locked before confirmation evaluation. Read ROUND18_PROTOCOL.md. All120 models
must have full training/NPZ acceptance before an evaluation manifest is generated.
No old/original checkpoints are included in this new confirmation factorial.

The manifest enumerates dataset0..4, recipe log/mixed crossed with short/long,
Frozen/Updated, model seed70/71/72, wind0/4.9/6.1/8.5 and episode0..9.
It binds checkpoint SHA256, resolved config, provenance, normalizer/split hashes,
training acceptance hashes, source hashes and both protocol document hashes.
Closed-loop environment/trajectory/planning seeds are1830000/1840000/1850000+e.
Each episode reuses unchanged round11_closed_loop.run at1500 scored steps.

Query40 shared first-decision anchors reproduce the100-step warmup exactly,
sample the64-candidate bank using the planning seed, and reuse the existing
round12 diagnostic engine. Query noise uses stage a1, index180000+anchor_index,
step100 and disjoint construct/evaluate phase words, with8 physical noise draws.
Full weighted returned solutions and all64 candidates are retained. New per-
dataset normalizers retain the original metric definition, never test-fitted.

Independent evaluation pilots: query4 anchors (episode0, all4 winds, all120models)
and control32 episodes (dataset0seed70, all8 factorial arms, episode0, all4winds).
Pilot outputs are separate and excluded from formal. Gates check all source,
checkpoint, actions/errors, physical reconstruction, warmup, candidate banks,
softmin costs/weights/solutions, and failure-aware scores at original tolerances.
Maximum pilot episode runtime*10 must be below2880s for10-episode formal tasks;
query maximum below720s. Only validated pilots allow each formal panel.

Formal:40 query anchors (4800 model records),4800 feedback episodes. Primary
statistical procedure and all directional criteria remain as in ROUND18_PROTOCOL.
Independent compact reconstruction verifies all arm means and51 offline,
204 query and68 control contrasts. Nonfinite query outcomes cannot be discarded
into a finite-only balanced factorial; preserve them and report incompleteness.
