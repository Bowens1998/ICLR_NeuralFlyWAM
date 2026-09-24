# Round15 evaluation implementation addendum

Fixed before inspection of trained outcomes. Preserves ROUND15_PROTOCOL.md bytes.
Fresh first-decision query indices0..29 enumerate wind4.9/6.1/8.5 ×episode0..9.
Query noise uses unchanged Round12 algorithm with stage=a1 and seed index150000+i:
[12012026,1,150000+i,100,phase], phase1construction/2evaluation; this is disjoint
from old query noise. First-decision state/history/nominal/bank use the new
610000/620000/630000environment/trajectory/planning seed families already specified.
No old donor trajectories or old trained models are needed for the primary test.

All60formal accepted models are evaluated per anchor. Control1800rows order:
dataset0..4,coverage log/mixed,memory frozen/updated,seed70..72,wind4.9/6.1/8.5,
episode0..9. Query pilot indices0/10/20. Control pilot all12rows for dataset0,
seed70,episode0 across4arms×3winds. Pilots remain separate and excluded.
Failures are retained; nonfinite query predictions have explicit status and
cannot be silently dropped from a claimed complete finite factorial. Closed-loop
failures use unchanged Round11failure-aware score and are never discarded.
Source/config/checkpoint/normalizer/protocol hashes, original model counts and
fresh seed mappings are verified before evaluation; every returned solution and
reported physical cost is recalculated from saved traces at acceptance.

Primary contrast and five-dataset uncertainty are as in the parent protocol.
Attribution remains to the controlled50%candidate-action training treatment,
including induced future state coverage, not a proven internal memory mechanism.

Execution: scripts/evidence/round15_submit.py renders/submits exclusive-claim arrays.
Query uses CPU, control uses RTX. Formal control batches10episodes/task across
180tasks (16concurrent); pilot acceptance must demonstrate10timesworst observed
episode duration below80%of1hour. Query pilot must fit80%of15minutes. These are
resource gates only, never outcome-selection gates. Recheck account resources
before submissions. All60trainingmodels must be accepted before manifest creation.
