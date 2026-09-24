"""Render accepted Round11D summaries; no new inference or episode selection."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
r = json.loads((ROOT/'reports/ROUND11D_RESULTS.json').read_text())
assert r['accepted_episodes'] == 2560
winds = ['0.0', '4.9', '6.1', '8.5']
x = np.arange(4)  # Wind conditions are categories, as in the original figure.
INK, MUTED, GRID = '#243445', '#536575', '#DCE3E8'
plt.rcParams.update({'font.family':'DejaVu Sans', 'pdf.fonttype':42,
    'ps.fonttype':42, 'svg.fonttype':'none', 'svg.hashsalt':'flight-memory-control',
    'font.size':14, 'axes.titlesize':16, 'axes.labelsize':14,
    'xtick.labelsize':13, 'ytick.labelsize':13, 'text.color':INK,
    'axes.labelcolor':INK, 'axes.edgecolor':GRID,
    'xtick.color':MUTED, 'ytick.color':MUTED})
fig = plt.figure(figsize=(11, 6.2))
ax = fig.add_axes([.07,.285,.40,.61])
bx = fig.add_axes([.61,.285,.37,.61])
fig.text(.014,.95,'A  Controller means',fontsize=16,weight='bold')
fig.text(.55,.95,'B  LN-by-memory interaction',fontsize=16,weight='bold')
for arm,label,color,style,marker in [
    ('frozen_r0','Frozen (no LN)','#0072B2','-','o'),
    ('frozen_r1','Frozen (LN)','#0072B2','--','s'),
    ('updated_r0','Updated (no LN)','#D55E00','-','o'),
    ('updated_r1','Updated (LN)','#D55E00','--','s')]:
    values=np.array([r['dataset_scores'][w]['capped_rmse_m'][arm] for w in winds])
    ax.plot(x,values.mean(axis=1),marker=marker,linestyle=style,color=color,
            label=label,ms=6,lw=1.9,mfc='white' if 'r1' in arm else color,zorder=3)
for kind,label,style,color in [
    ('nominal','Nominal-physics MPPI',':',INK),
    ('true_mean','True-mean-wind MPPI','-.','#307E70'),
    ('pd','PD',':','#ADB3BA'),
    ('preview_pd','Preview PD','-.','#9D8969')]:
    ax.plot(x,[r['references'][w][kind]['capped_rmse_m'] for w in winds],
            linestyle=style,color=color,label=label,lw=1.6,zorder=2)
ax.set(ylabel='Tracking RMSE (m)',ylim=(0,.75))
fig.legend(*ax.get_legend_handles_labels(),fontsize=12,ncol=4,
           loc='lower center',bbox_to_anchor=(.5,.015),frameon=False,
           columnspacing=1.2,handletextpad=.5,handlelength=2.2,labelspacing=.65)
for i,w in enumerate(winds):
    c=r['contrasts'][w]['capped_rmse_m']['recurrence_effect_updated_minus_frozen']
    lo,hi=c['descriptive_t95ci']
    bx.vlines(i,lo,hi,color=INK,lw=1.6,zorder=2)
    bx.scatter(np.full(5,i)+np.linspace(-.09,.09,5),c['differences'],
               color='#9AA9B5',s=32,zorder=3)
    bx.scatter(i,c['mean'],marker='D',color=INK,s=65,zorder=4)
bx.axhline(0,color=MUTED,ls='--',lw=.9)
bx.set(ylabel='LN-by-memory interaction (m)')
bx.text(.025,.97,'Dots: datasets   ◆: means\nBars: descriptive 95% intervals',
        transform=bx.transAxes,va='top',fontsize=12,color=MUTED,linespacing=1.5)
for axis in (ax,bx):
    axis.set_xticks(x,['0','4.9','6.1','8.5'])
    axis.set_xlabel('Static wind speed (m/s)',labelpad=7)
    axis.spines[['top','right']].set_visible(False)
    axis.grid(axis='y',color=GRID,lw=.65,zorder=0)
    axis.set_axisbelow(True)
for ext in ('pdf','svg','png'):
    fig.savefig(ROOT/f'paper/figures/round11d_fair_closed_loop.{ext}',dpi=180,facecolor='white')
svg = ROOT/'paper/figures/round11d_fair_closed_loop.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
plt.close(fig)
