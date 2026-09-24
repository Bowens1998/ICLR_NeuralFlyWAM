"""Generate post-review diagnostic tables from accepted, paired summaries."""
import json
from pathlib import Path
R=Path(__file__).resolve().parents[2]
r=json.loads((R/'reports/ROUND13_EXISTING_RESULTS.json').read_text())
b=json.loads((R/'reports/ROUND13_QUERY_BOUNDARIES.json').read_text())
s=[r'''\clearpage
\section{Paired reversal and query boundaries}
\label{app:round13}
These post-review analyses use the existing accepted checkpoints and traces.
They are descriptive and result-informed; no new significance tests are added.
The independent units remain five training datasets, not individual windows,
candidates, seeds or entries of a dataset-by-episode matrix.

\paragraph{Joint reversal and evaluation instances.}
At 6.1 and 8.5\,m/s, respectively, 12/15 and 13/15 paired model-seed comparisons
improve logged prediction while worsening control. Aggregating seeds first gives
4/5 datasets at each wind (Figure~\ref{fig:rankings}); dataset D1 is the exception
in both. After averaging the fixed trained roster, all ten original evaluation
episodes have positive control differences at both winds. These episode summaries
are conditional on those trained models, not additional independent training
replicates. Full seed pairs and dataset-by-episode matrices accompany the numerical
supplement.

\paragraph{Horizon and physical-state components.}
We recompute query errors from the saved model predictions and the independent
physical evaluation-noise stream, checking agreement with every original terminal
metric before stratifying. Table~\ref{tab:r13horizon} retains all four winds and
five pre-existing horizons. The aggregate gap is already positive at step 1,
primarily through angular rate. This is compatible with different trained weights;
it cannot identify drift caused by repeated memory updates, since identical
weights give identical first predictions. At both shared held-out winds, physical
position differences remain negative at step 10 but turn positive by step 25.
Thus the logged-action advantage fails early for one component, while position
consequences develop later; no unique internal mechanism is established.

\begin{table}[htbp]\centering\small
\caption{Terminal query errors, updated minus frozen without LN. All are means
across five datasets. $\Delta v$, $\Delta R$, $\Delta\omega$ use the original
normalized errors; $\Delta p$ is physical position error in meters.}
\label{tab:r13horizon}
\begin{tabular}{rrrrrrr}\toprule
Wind & Step & $\Delta E$ & $\Delta v$ & $\Delta R$ & $\Delta\omega$ & $\Delta p$\\\midrule''']
for w,hs in r['a1_terminal_horizon'].items():
 for h,m in hs.items():
  v=[m[k]['mean'] for k in ['original_E','original_velocity_E','original_orientation_E','original_angular_rate_E','physical_position_l2_m']]
  s.append(w+' & '+h+' & '+' & '.join(f'{x:.5f}' for x in v)+r' \\')
s.append(r'''\bottomrule\end{tabular}\end{table}
\clearpage
\paragraph{Candidate amplitude.}
For perturbation $\epsilon\in\mathbb R^{50\times3}$, define
$a=\sqrt{\operatorname{mean}[(\epsilon/\sigma)^2]}$ before action clipping,
where $\sigma=(0.04,0.08,0.08)$. Fixed bins separate zero, $(0,0.5)$,
$[0.5,1)$ and $[1,\infty)$. Per wind, the ten anchors contain 20 zero candidates,
496 in $(0,0.5)$, 124 in $[0.5,1)$ and none in $[1,\infty)$.
Two candidates coincide at zero at each first decision (nominal and empty warm
start); both are retained to match the actual planner bank. Each nonempty bin
contains candidates at all ten anchors. We average candidates and evaluation-noise
draws within anchor, then equally average anchors and seeds within dataset.
Across every populated bin, the mean terminal aggregate gap is positive at all
five registered horizons, including zero perturbation. This does not identify
a large-action threshold; larger amplitudes are absent from this first-decision
sample. Bins are observational subsets, not randomized interventions.

\begin{table}[htbp]\centering\small
\caption{Amplitude-stratified terminal errors at step 50, no LN. Intervals are
paired dataset-level descriptive 95\% intervals for $\Delta E$; all empty bins
are reported explicitly above. Full horizons/components remain in the supplement.}
\begin{tabular}{rlrrl}\toprule
Wind & Amplitude & $\Delta E$ & $\Delta p$ (m) & $\Delta E$ interval\\\midrule''')
for w,bins in b['wind_strata'].items():
 for label,v in bins.items():
  if not v['total_candidates']:continue
  m=v['terminal']['50'];e=m['original_E'];p=m['physical_position_l2_m']['mean'];lo,hi=e['descriptive_t95ci']
  s.append(f'{w} & {label} & {e["mean"]:.4f} & {p:.4f} & [{lo:.4f}, {hi:.4f}]'+r' \\')
s.append(r'''\bottomrule\end{tabular}\end{table}

\paragraph{Finite-noise precision.}
The paired first-decision physical solution-cost differences at winds
0/4.9/6.1/8.5\,m/s are 0.3610/1.2132/1.7228/2.6318; their Monte Carlo standard
errors are 0.00379/0.01134/0.01311/0.02355. This calculation preserves shared
noise across models and treats the ten independently seeded anchors as noise
units. It quantifies evaluation-noise uncertainty conditional on the constructed
solutions; it does not include oracle-construction uncertainty or replace the
five-dataset intervals. Later-query donor/time strata remain distinct in
Appendix~\ref{app:round12}, including adverse and uncertain cases.
''')
(R/'paper/round13_appendix.tex').write_text('\n'.join(s)+'\n')

fresh_path=R/'reports/ROUND13_FRESH_RESULTS.json'
if fresh_path.exists():
 f=json.loads(fresh_path.read_text())
 fresh=[r'''\clearpage
\section{New closed-loop evaluation instances}
\label{app:fresh}
After the initial results and external feedback, we fixed a new evaluation panel
before inspecting its outcomes. All 30 original no-LN GRU checkpoints were retained
(five datasets, three training seeds and two memory schedules), without retraining,
model selection or controller changes. Each model ran 20 new episodes at each
shared held-out wind, totaling 1,200 episodes. Episode $e=0,\ldots,19$ uses trajectory
seed $430000+e$, environment-noise seed $440000+e$ and planning seed $450000+e$,
paired across models and winds. These seeds do not overlap the original panel.
The plant, trajectory generator, H50/N64 planner, 100-step PD warmup, 1,500 scored
steps and failure caps are unchanged. Four full-length pilot episodes used a
separate namespace and never enter these summaries. All formal episodes passed
provenance, action-trace, common-warmup and scoring checks; zero numerical failures
occurred. Runs used RTX PRO 6000 GPUs with the existing CUDA evaluation environment.

\begin{table}[htbp]\centering\small
\caption{New-trajectory no-LN control results. RMSE and differences are in meters.
Dataset intervals average three training seeds and 20 episodes within each of five
datasets. Episode intervals average the fixed 15 trained models per arm and are
conditional on that roster. Neither interval treats 100 dataset--episode cells as
independent replicates.}
\begin{tabular}{r r r r l l}\toprule
Wind & Frozen & Updated & $\Delta$ & Dataset 95\% interval & Episode 95\% interval\\\midrule''']
 for w,v in f['wind_strata'].items():
  d=v['dataset_contrast'];e=v['episode_contrast_conditional_on_trained_roster']
  fresh.append(f"{w} & {v['frozen_mean']:.5f} & {v['updated_mean']:.5f} & {d['mean']:.5f} & [{d['descriptive_t95ci'][0]:.5f}, {d['descriptive_t95ci'][1]:.5f}] & [{e['descriptive_t95ci'][0]:.5f}, {e['descriptive_t95ci'][1]:.5f}]"+r' \\')
 fresh.append(r'''\bottomrule\end{tabular}\end{table}

\begin{table}[htbp]\centering\small
\caption{All five training-dataset differences on new trajectories: updated minus
frozen RMSE (m). Dataset labels match Figure~\ref{fig:rankings}.}
\begin{tabular}{rrrrrr}\toprule
Wind & D1 & D2 & D3 & D4 & D5\\\midrule''')
 for w,v in f['wind_strata'].items():fresh.append(w+' & '+' & '.join(f'{x:.5f}' for x in v['dataset_contrast']['values'])+r' \\')
 fresh.append(r'''\bottomrule\end{tabular}\end{table}

The updated-minus-frozen difference is positive for 4/5 datasets at 6.1\,m/s
and 5/5 at 8.5\,m/s. The original exception D1 still slightly favors updating
at 6.1\,m/s, and slightly favors freezing at 8.5\,m/s on the new panel. After
averaging the fixed trained roster, all 20 individual episode contrasts are
positive at each wind. This does not mean every model wins on every episode.
The supplement retains the complete $5\times20$ dataset--episode matrices,
three-seed summaries and all per-arm outcomes; old and new episodes are not pooled.

The fresh panel supports persistence of the mean control ordering across new
trajectory and noise draws from this generator. It does not add independent
training datasets, simulators, robot platforms or physical trials, and does not
establish a universal memory-design principle. No additional hypothesis tests
or outcome-dependent seed expansion were performed.
''')
 (R/'paper/round13_fresh_appendix.tex').write_text('\n'.join(fresh)+'\n')
