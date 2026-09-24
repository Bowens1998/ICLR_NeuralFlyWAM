"""Generate the targeted-review appendix directly from accepted summaries."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(n):return json.loads((ROOT/f'reports/ROUND14_{n}.json').read_text())
def ci(d):
 lo,hi=d['descriptive_t95ci'];return f"${d['mean']:+.5f}$ & $[{lo:+.5f},{hi:+.5f}]$"
def table(caption,label,cols,header,rows):
 return '\n'.join([r'\begin{table}[htbp]',r'\centering\small',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+cols+'}',r'\toprule',header+r'\\',r'\midrule',*rows,r'\bottomrule',r'\end{tabular}',r'\end{table}'])+'\n'
s=r'''\clearpage
\section{Targeted action-source and context-access tests}
\label{app:round14}
These result-informed follow-ups address specific ambiguities in the original
comparison. All use the five independently generated training datasets as the
uncertainty unit, averaging three paired model seeds and flights or anchors
within each dataset. Intervals are descriptive paired Student-$t$ 95\% intervals
with four degrees of freedom, not simultaneous intervals. No new p-values,
outcome-selected seeds, wind thresholds or pooled old/new evaluation samples
are introduced. Formal manifests were fixed before their evaluations; development
pilots are excluded. The complete machine-readable summaries retain individual
dataset differences and component/horizon results.

\subsection{Offline components and physical position}
We audit the 30 original no-LN models and 30 new auxiliary-context models on
all static test windows. Each model is evaluated in the original 85,500-window
order with batches of 256, retaining the 57,000 windows at the two primary winds
(2,850 per flight, ten flights per wind). Thus the position audit contains
3,420,000 model--window pairs, not independent experimental replicates.
For $h=1,\ldots,50$,
\[
\hat p_h=p_0+0.02\sum_{j=1}^{h}\hat v_j,\qquad
 e_{p,h}=\|\hat p_h-p_h^{\rm recorded}\|_2.
\]
The prefix metric averages these physical-meter errors over all 50 steps;
it is distinct from terminal error and from the previously normalized
velocity-integral metric. A secondary calculation integrates recorded velocities
to separate the recorded-position target from the numerical integration convention.

The original normalized velocity, rotation, rate, aggregate and displacement
arrays are reproduced exactly for every model (maximum absolute difference zero).
An original-model pilot on RTX hardware failed the locked orientation-reproduction
gate; rerunning that model unchanged on its original L4 hardware passed exactly.
The failed artifact is retained. Formal original models use L4 and new models
use RTX, with paired arms on the same hardware within each family. Tolerances
were not relaxed. This verifies the accepted predictions on their evaluation
hardware, rather than demonstrating hardware-invariant numerics.
'''
p=read('POSITION_RESULTS');o=read('OFFLINE_COMPONENTS');rows=[]
for w in ['6.1','8.5']:
 for metric,title in [('physical_position_l2_m/prefix_mean','Mean over 50 steps'),('physical_position_l2_m/h50','Terminal step 50')]:
  v=p['summary'][w][metric];m=v['arm_means'];rows.append(f'{w} & {title} & {m["frozen"]:.5f} & {m["updated"]:.5f} & '+ci(v['contrasts']['updated_minus_frozen'])+r'\\')
s+=table('Original-model physical-position errors in meters. Differences are Updated minus Frozen; both prefix and terminal comparisons favor Updated.','tab:r14position','llrrrr',r'Wind & Outcome & Frozen & Updated & Difference & 95\% interval',rows)
rows=[]
for w,key in [('6.1','static_50wind'),('8.5','static_70wind')]:
 for h in ['1','5','10','25','50','prefix_mean']:
  vals=[o['summary'][key][c][h]['mean'] for c in ['velocity','orientation','angular_rate','aggregate']]
  rows.append(w+' & '+('Mean 1--50' if h=='prefix_mean' else h)+' & '+' & '.join(f'${v:+.5f}$' for v in vals)+r'\\')
s+=table('Original no-LN Updated minus Frozen normalized component errors. Numbered rows use the error at that step, not the prefix average. Early component exceptions are retained. Dataset values and intervals are in the accompanying numerical records.','tab:r14components','llrrrr',r'Wind & Step & Velocity & Rotation & Angular rate & $E$',rows)
s+=r'''
\subsection{Changing only future action source}
At step 100 of each of ten static test flights in each dataset and primary wind,
we retain the complete observed history and physical simulator state. Replay
through step 150 is checked against the original CSV fields with a $5.1\times
10^{-9}$ absolute tolerance for their eight-decimal serialization. The action
comparison uses one logged sequence versus a fresh 64-candidate MPPI bank,
initialized from the original PD plan with empty warm start. This is a newly
constructed first-decision bank, not a claim to recover a historical controller's
bank. The physical branches share eight paired noise samples; both sources use
float32 actions and 10 integration substeps per control step. All six original
no-LN models from the corresponding dataset evaluate each anchor, totaling
100 anchors and 600 model records. No anchor is omitted.

Both the logged-action and candidate-action branches are evaluated under the
same fresh physical noise. The actual realized logged future is an additional
check, not substituted for the paired logged-action branch. Model errors
use the original training-only scales and Equation~\ref{eq:metric}.
'''
a=read('ACTION_RESULTS');rows=[]
for w in ['6.1','8.5']:
 for k,title in [('logged','Logged actions, paired noise'),('candidate','Candidate actions, paired noise'),('actual_logged_future','Actual logged future'),('candidate_minus_logged_interaction','Candidate minus logged contrast')]:
  rows.append(w+' & '+title+' & '+ci(a['summary'][w]['h50/original_E/prefix_mean'][k])+r'\\')
s+=table('Action-source experiment: Updated minus Frozen H50 prefix $E$. The final row at each wind is a difference of those paired differences.','tab:r14actions','llrr',r'Wind & Source / contrast & Difference & 95\% interval',rows)
s+=r'''
This test removes history, initial-state and physical-noise changes as explanations
of the action-source contrast. At 6.1\,m/s the logged-action interval crosses
zero; it is the relative deterioration on candidates that is consistent across
all five datasets. The experiment does not separate the bank's nominal action
sequence from its perturbations, identify an internal latent mechanism, or
quantify how much of the tracking RMSE gap it explains.

\subsection{Auxiliary context with dynamic rollout memory}
Equation~\ref{eq:access} adds a 64-dimensional auxiliary input to the original
GRUCell. Both arms update the rollout state; only $q_h=m_0$ versus $q_h=m_h$
differs. The history encoder, width, output head, physical integrator, losses,
data, ID checkpoint rule and maximum 120-epoch training recipe remain shared.
All five datasets and seeds 70--72 are retained, with separate training for
each arm. Both have 51,785 trainable parameters, 12,288 more than the original
family. Paired initialization gives identical first-step output at identical
weights; separately trained weights need not do so. All auxiliary-input weights
are active in gradient computations. Parameter-count matching does not equate
effective function classes, because the dynamic input duplicates hidden state
in some gate transformations. Comparison with original Frozen/Updated models
changes architecture and capacity, not only context access.

Query evaluation reuses the original 20 first-decision anchors at 6.1/8.5\,m/s,
with identical snapshots, candidate banks and construction/evaluation noise.
All 600 new model--anchor records are accepted. Physical candidate costs agree
with the original traces to $10^{-12}$; the original saved no-LN metrics are
regenerated before accepting new results. Both new arms run in CPU FP32 for
this query contrast; cross-family numerical comparisons are not a hardware test.
The actual softmin-weighted sequence is independently simulated with eight
noise samples, as in Appendix~\ref{app:round12}.

Closed-loop evaluation reuses the original ten episodes per wind and unchanged
50-step/64-candidate controller: 30 models $\times$ two winds $\times$ ten
episodes gives 600 accepted episodes, with zero numerical or threshold failures.
The 100-step PD warmup, 1,500 scored steps and failure-aware scoring are unchanged.
No original-model episodes are rerun or duplicated as independent observations.
'''
r=read('ACCESS_PREDICTION');q=read('QUERY_RESULTS');c=read('CONTROL_RESULTS');rows=[]
for w in ['6.1','8.5']:
 metrics=[]
 v=r['summary'][w]['aggregate']['prefix_mean'];metrics.append(('Logged $E$',{k:sum(x)/len(x) for k,x in v['dataset_scores'].items()},v['fixed_minus_dynamic']))
 v=p['summary'][w]['physical_position_l2_m/prefix_mean'];metrics.append(('Logged position (m)',v['arm_means'],v['contrasts']['fixed_minus_dynamic']))
 key=next(k for k in q['summary'][w] if 'original_E' in k and 'h50' in k and 'prefix_mean' in k)
 v=q['summary'][w][key];metrics.append(('Candidate $E$',v['arm_means'],v['contrasts']['fixed_minus_dynamic']))
 v=q['summary'][w]['physical_solution_cost'];metrics.append(('Physical solution cost',v['arm_means'],v['contrasts']['fixed_minus_dynamic']))
 v=c['summary'][w];metrics.append(('Tracking RMSE (m)',v['arm_means'],v['fixed_minus_dynamic']))
 for title,m,d in metrics:rows.append(f'{w} & {title} & {m["fixed"]:.5f} & {m["dynamic"]:.5f} & '+ci(d)+r'\\')
s+=table('Auxiliary-context comparison. Differences are fixed minus dynamic auxiliary context; both arms evolve their rollout state. All metrics are errors or costs (lower is better), but their units differ.','tab:r14access','llrrrr',r'Wind & Outcome & Fixed aux. & Dynamic aux. & Difference & 95\% interval',rows)
s+=r'''
Persistent access improves common-query prediction and physical solution cost
in all five datasets at both winds. Tracking RMSE improves in 5/5 datasets at
6.1\,m/s and 4/5 at 8.5\,m/s, where the interval crosses zero. It worsens logged
physical-position prediction in every dataset at both winds. Original Frozen
models still have lower mean tracking RMSE (0.210/0.368\,m) than either new arm.
These mixed results support evaluating context access under candidate actions,
not claiming that the hybrid dominates or that improved offline position accuracy
is necessary for improved tracking. The secondary 3.7\,m/s logged $E$ difference
is $-0.00197$ ($[-0.00352,-0.00042]$); this interpolation condition was not added
to the primary query/control tests after observing results.
'''
(ROOT/'paper').mkdir(parents=True,exist_ok=True)
(ROOT/'paper/round14_appendix.tex').write_text(s)
