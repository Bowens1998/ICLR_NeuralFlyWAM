"""Render the accepted evidence chain, preserving every plotted statistic.

Panel A: paired dataset differences. B: fixed-anchor action-source comparison.
C: all four recipes and both budgets, on identical scales at the two winds.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
def read(n):
    return json.loads((ROOT/'reports'/n).read_text())
r = read('ROUND13_EXISTING_RESULTS.json')
a = read('ROUND14_ACTION_RESULTS.json')
c = read('ROUND16_CONTROL_RESULTS.json')
INK, MUTED, GRID = '#243445', '#536575', '#DCE3E8'
# Wind colors are distinct from the blue/orange memory-design colors.
cols = ['#307E70', '#7963A6']
winds = ['6.1', '8.5']
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':14,
    'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none',
    'svg.hashsalt':'flight-memory-evidence', 'text.color':INK,
    'axes.labelcolor':INK, 'xtick.color':MUTED, 'ytick.color':MUTED,
    'axes.edgecolor':GRID, 'axes.labelsize':14, 'xtick.labelsize':13,
    'ytick.labelsize':13})
fig = plt.figure(figsize=(11, 6.3))
ax = fig.add_axes([.075,.575,.36,.335])
bx = fig.add_axes([.635,.575,.35,.335])
cx = fig.add_axes([.16,.115,.325,.26])
dx = fig.add_axes([.66,.115,.325,.26])
fig.text(.014,.969,'A  Same models, different rankings',size=16,weight='bold')
fig.text(.525,.969,'B  Change future actions only',size=16,weight='bold')
fig.text(.014,.445,'C  Training recipe × budget',size=16,weight='bold')
for axis in (ax,bx,cx,dx):
    axis.spines[['top','right']].set_visible(False)
    axis.tick_params(length=3, width=.7)
    axis.grid(axis='x', color=GRID, lw=.6, zorder=0)
    axis.set_axisbelow(True)
ax.axhline(0,color=MUTED,lw=.9,zorder=1)
ax.axvline(0,color=MUTED,lw=.9,zorder=1)
for w,col in zip(winds,cols):
    v=r['joint'][w]
    ax.scatter([p['prediction'] for p in v['pairs']],
               [p['control'] for p in v['pairs']],s=39,color=col,alpha=.8,zorder=3)
    ax.scatter(v['prediction']['mean'],v['control']['mean'],s=65,
               color=col,marker='D',edgecolor=INK,lw=.8,zorder=4)
ax.set(xlabel=r'Logged $\Delta E$ (U − F)',ylabel='Tracking ΔRMSE (m; U − F)')
ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
ax.legend(handles=[Line2D([],[],color=col,marker='o',ls='',label=f'{w} m/s')
                   for w,col in zip(winds,cols)],frameon=False,fontsize=12,
          loc='upper left',handletextpad=.3,borderaxespad=.2)

bx.axvline(0,color=MUTED,lw=.9,zorder=1)
for wi,(w,col) in enumerate(zip(winds,cols)):
    for si,source in enumerate(['logged','candidate']):
        y=3-wi*2-si
        d=a['summary'][w]['h50/original_E/prefix_mean'][source]
        lo,hi=d['descriptive_t95ci'];m=d['mean']
        bx.plot([lo,hi],[y,y],color=col,lw=2.1,zorder=2)
        bx.scatter(d['dataset_differences'],y+np.linspace(-.13,.13,5),
                   color=col,s=25,alpha=.5,zorder=3)
        bx.scatter(m,y,color=col,marker='D',edgecolor=INK,lw=.7,s=56,zorder=4)
    bx.plot([a['summary'][w]['h50/original_E/prefix_mean'][s]['mean']
             for s in ['logged','candidate']], [3-wi*2,2-wi*2],color=col,ls=':',lw=1.1)
bx.set(xlabel=r'Prediction $\Delta E$ (U − F)',yticks=[3,2,1,0],
       yticklabels=['6.1: logged','6.1: candidate','8.5: logged','8.5: candidate'],
       ylim=(-.4,3.4),xlim=(-.04,.25),xticks=[0,.1,.2])
bx.tick_params(axis='y',length=0)

for axis,w,col in [(cx,'6.1',cols[0]),(dx,'8.5',cols[1])]:
    axis.axvline(0,color=MUTED,lw=.9,zorder=1)
    for ri,recipe in enumerate(['dense','recorded','log','mixed']):
        for budget,offset,marker in [('short',.16,'o'),('long',-.16,'D')]:
            d=c['summary']['rmse_m'][w]['contrasts'][f'{recipe}_{budget}/U-F']
            m=d['mean'];lo,hi=d['descriptive_t95ci'];y=3-ri+offset
            axis.plot([lo,hi],[y,y],color=col,lw=1.7,alpha=.5 if budget=='short' else 1)
            axis.scatter(m,y,facecolor='white' if budget=='short' else col,
                         edgecolor=col,marker=marker,s=42,lw=1.4,zorder=4)
    axis.set(yticks=[3,2,1,0],yticklabels=['Dense','Repeated','Re-simulated','Mixed'],
             ylim=(-.5,3.5),xlim=(-.43,.30),xticks=[-.4,-.2,0,.2],
             xlabel='Tracking ΔRMSE (m; U − F)')
    axis.tick_params(axis='y',length=0)
    axis.set_title(f'{w} m/s',fontsize=14,color=col,weight='bold',pad=6)
fig.legend(handles=[Line2D([],[],color=INK,marker='o',mfc='white',ls='',label='Short budget'),
                    Line2D([],[],color=INK,marker='D',ls='',label='Long budget')],
           loc='upper right',bbox_to_anchor=(.988,.493),ncol=2,fontsize=12,
           frameon=False,handletextpad=.35,columnspacing=1)
fig.text(.5,.013,'U − F < 0: Updated has lower error     |     U − F > 0: Frozen has lower error',
         ha='center',fontsize=12,color=MUTED)
for ext in ('pdf','svg','png'):
    fig.savefig(ROOT/f'paper/figures/round16_evidence_chain.{ext}',dpi=180,facecolor='white')
svg = ROOT/'paper/figures/round16_evidence_chain.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
plt.close(fig)
