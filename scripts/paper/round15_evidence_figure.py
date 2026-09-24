"""Main evidence chain; every point and interval comes from accepted summaries."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
def read(n):return json.loads((ROOT/'reports'/n).read_text())
r=read('ROUND13_EXISTING_RESULTS.json');a=read('ROUND14_ACTION_RESULTS.json');q=read('ROUND15_QUERY_RESULTS.json');c=read('ROUND15_CONTROL_RESULTS.json')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.titlesize':8,'axes.labelsize':7})
fig=plt.figure(figsize=(6.5,3.6),layout='constrained');gs=fig.add_gridspec(2,3,width_ratios=[1,1,1.05])
ax=fig.add_subplot(gs[:,0]);bx=fig.add_subplot(gs[:,1]);cx=fig.add_subplot(gs[0,2]);dx=fig.add_subplot(gs[1,2])
cols=['#0072B2','#D55E00'];winds=['6.1','8.5']
ax.axhline(0,color='.65',lw=.7);ax.axvline(0,color='.65',lw=.7)
for w,col in zip(winds,cols):
 v=r['joint'][w];ax.scatter([p['prediction'] for p in v['pairs']],[p['control'] for p in v['pairs']],s=19,color=col,alpha=.85,label=f'{w} m/s')
 ax.scatter(v['prediction']['mean'],v['control']['mean'],s=29,color=col,marker='D',edgecolor='black',lw=.5)
ax.set(title='A  Same models, different rankings',xlabel=r'Logged $\Delta E$ (U − F)',ylabel='Tracking ΔRMSE (m; U − F)');ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0));ax.legend(frameon=False,loc='upper right',fontsize=7)
bx.axvline(0,color='.65',lw=.7)
for wi,(w,col) in enumerate(zip(winds,cols)):
 for si,source in enumerate(['logged','candidate']):
  y=3-wi*2-si;d=a['summary'][w]['h50/original_E/prefix_mean'][source];lo,hi=d['descriptive_t95ci'];m=d['mean']
  bx.plot([lo,hi],[y,y],color=col,lw=1.4);bx.scatter(d['dataset_differences'],y+np.linspace(-.11,.11,5),color=col,s=10,alpha=.6);bx.scatter(m,y,color=col,marker='D',s=24,zorder=5)
 bx.plot([a['summary'][w]['h50/original_E/prefix_mean'][s]['mean'] for s in ['logged','candidate']],[3-wi*2,2-wi*2],color=col,ls=':',lw=.8)
bx.set(title='B  Change future actions only',xlabel=r'Prediction $\Delta E$ (U − F)',yticks=[3,2,1,0],yticklabels=['6.1: log','6.1: bank','8.5: log','8.5: bank'],ylim=(-.45,3.5));bx.tick_params(axis='y',length=0)
for axis,key,title,xlabel in [(cx,'query','C  Change training coverage','Δ(U − F): solution cost'),(dx,'control',None,'Δ(U − F): tracking RMSE (m)')]:
 axis.axvline(0,color='.65',lw=.7)
 for wi,(w,col) in enumerate(zip(winds,cols)):
  d=q['summary']['physical_solution_cost'][w]['contrasts']['coverage_by_memory_interaction'] if key=='query' else c['summary']['rmse_m'][w]['contrasts']['coverage_by_memory_interaction'];m=d['mean'];lo,hi=d['descriptive_t95ci'];y=1-wi
  axis.plot([lo,hi],[y,y],color=col,lw=1.4);axis.scatter(d['dataset_differences'],y+np.linspace(-.1,.1,5),color=col,s=10,alpha=.6);axis.scatter(m,y,color=col,marker='D',s=24,zorder=5)
 axis.set(xlabel=xlabel,yticks=[1,0],yticklabels=['6.1','8.5'],ylim=(-.4,1.4));axis.tick_params(axis='y',length=0)
 if title:axis.set_title(title)
for axis in [ax,bx,cx,dx]:axis.spines[['top','right']].set_visible(False)
for ext in ['pdf','svg']:fig.savefig(ROOT/f'paper/figures/round15_evidence_chain.{ext}')
