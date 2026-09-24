"""Editable paired-dataset figure generated only from accepted descriptive results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
r=json.loads((ROOT/'reports/ROUND13_EXISTING_RESULTS.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axs=plt.subplots(1,2,figsize=(6.5,2.5),layout='constrained')
colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
for ax,w in zip(axs,['6.1','8.5']):
    v=r['joint'][w]
    ax.axhline(0,color='.45',lw=.8);ax.axvline(0,color='.45',lw=.8)
    for i,p in enumerate(v['pairs']):
        ax.scatter(p['prediction'],p['control'],s=33,color=colors[i],zorder=3)
        ax.annotate(f'D{i+1}',(p['prediction'],p['control']),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.scatter(v['prediction']['mean'],v['control']['mean'],marker='D',s=38,color='black',label='Mean',zorder=4)
    ax.set(title=f'{w} m/s: joint reversal in {v["joint_reversal_count"]}/5 datasets',xlabel=r'$\Delta E$ (updated $-$ frozen)',ylabel=r'$\Delta$ tracking RMSE (m)')
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlim(min(p['prediction'] for p in v['pairs'])*1.22,.0015 if w=='6.1' else .003)
    ax.set_ylim(-.065,.26 if w=='6.1' else .43)
    ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    ax.legend(frameon=False,fontsize=8,loc='lower left')
for ext in ['pdf','svg']:
    fig.savefig(ROOT/f'paper/figures/round13_joint_reversal.{ext}')
