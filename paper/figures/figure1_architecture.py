"""Editable overview of the existing model comparison and evidence sequence.

Run with Python + Matplotlib. All labels and geometry remain vector objects.
Colors identify memory designs, not measured outcomes. No data are generated.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath

INK = '#243445'
MUTED = '#536575'
LINE = '#9AA9B5'
BLUE = '#0072B2'
ORANGE = '#D55E00'
PURPLE = '#7963A6'
GREEN = '#307E70'


def draw(output_dir=None):
    out = Path(output_dir) if output_dir else Path(__file__).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15,
                         'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'flight-memory-figure1'})
    fig = plt.figure(figsize=(11, 8.35))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1100), ylim=(835, 0)); ax.axis('off')
    bounded_text = []

    def text(x, y, label, size=15, color=INK, weight='normal', ha='left', region=None):
        t = ax.text(x, y, label, fontsize=size, color=color, weight=weight,
                    ha=ha, va='center', linespacing=1.05)
        if region is not None:
            bounded_text.append((t, region))
        return t

    def rect(x, y, w, h, fill='white', edge='#DCE3E8', radius=8, lw=1):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={radius}',
                           facecolor=fill, edgecolor=edge, lw=lw)
        ax.add_patch(p)
        return p

    def arrow(points, color=MUTED, lw=1.5, dashed=False):
        ax.add_patch(FancyArrowPatch(path=MplPath(points, [MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1)),
                      arrowstyle='-|>', mutation_scale=12, color=color, lw=lw,
                      linestyle=(0, (3, 3)) if dashed else '-', joinstyle='round', capstyle='round'))

    def dot(x, y, color=MUTED):
        ax.add_patch(Circle((x, y), 2.7, color=color, zorder=4))

    def heading(y, letter, title):
        rect(14, y-13, 28, 26, INK, INK, radius=4)
        text(28, y, letter, 15, 'white', 'bold', 'center')
        text(53, y, title, 18, weight='bold')

    # One-step DAG: state/memory at h enter; state/memory at h+1 leave.
    # Temporal recurrence is specified below, never drawn as an algebraic loop.
    heading(22, 'A', 'One observed history, two memory designs')
    rect(14, 52, 198, 384, '#F1F5F8')
    rect(242, 52, 844, 384, '#FBFCFD')
    text(113, 78, 'Observed history', 14.5, weight='bold', ha='center')
    text(260, 78, 'Imagined rollout', 16, weight='bold')
    text(1068, 78, '50 steps / 1 s · no new observations', 14, MUTED, ha='right')
    rect(30, 112, 166, 57)
    text(113, 127, 'History features', 15, ha='center')
    text(113, 156, '100 steps / 2 s', 14, MUTED, ha='center')
    arrow([(113, 169), (113, 199)])
    rect(30, 199, 166, 66, '#E5EDF3', '#A9B9C6')
    text(113, 220, 'GRU encoder', 16, weight='bold', ha='center')
    text(113, 247, '+ projection', 15, ha='center')
    text(113, 321, 'Re-encode at', 15, MUTED, ha='center')
    text(113, 352, 'each real step', 15, MUTED, ha='center')
    text(113, 400, 'Weights stay fixed', 14, MUTED, ha='center')

    # Initialization and Frozen both receive raw encoded history m0.
    arrow([(196, 232), (267, 232), (267, 192), (315, 192), (315, 213)])
    text(226, 210, r'$m_0$', 18, ha='center')
    text(315, 175, r'At $h=0$', 14, MUTED, ha='center')
    rect(279, 213, 72, 45, edge='#A9B9C6', radius=20)
    text(315, 235, r'$m_h$', 19, ha='center')
    arrow([(351, 235), (389, 235)])
    rect(389, 213, 63, 45, '#F0EBF6', '#BBAECE')
    text(420.5, 235, r'$N_r$', 19, PURPLE, ha='center')
    text(420.5, 288, 'Memory input', 14, MUTED, ha='center')
    arrow([(452, 235), (478, 235)])
    rect(478, 203, 128, 65, '#E5EDF3', '#8196A6')
    text(542, 220, 'GRUCell', 16, weight='bold', ha='center')
    text(542, 252, 'Hidden: 64', 14, MUTED, ha='center')

    # Predicted state goes to BOTH feature assembly and physical integration.
    rect(314, 100, 72, 36, edge='#A9B9C6', radius=18)
    text(350, 118, r'$\hat{s}_h$', 18, ha='center')
    arrow([(350, 136), (350, 151), (478, 151)])
    dot(386, 118)
    arrow([(386, 118), (1023, 118), (1023, 322)], LINE, 1.2)
    text(895, 145, 'Physical state', 14, MUTED, ha='center')
    rect(478, 130, 128, 43)
    text(542, 151, r'Features $x_h$', 15, ha='center')
    arrow([(542, 173), (542, 203)])
    rect(660, 133, 63, 36, edge='#A9B9C6', radius=18)
    text(691.5, 151, r'$a_h$', 18, ha='center')
    arrow([(660, 151), (606, 151)])

    arrow([(606, 235), (687, 235)])
    dot(642, 235, ORANGE)
    text(642, 210, r'$c_h$', 18, ha='center')
    rect(687, 213, 63, 45, '#F0EBF6', '#BBAECE')
    text(718.5, 235, r'$N_o$', 19, PURPLE, ha='center')
    text(718.5, 288, 'Readout', 14, MUTED, ha='center')
    arrow([(750, 235), (783, 235)])
    rect(783, 213, 132, 45)
    text(849, 235, 'Linear head', 15, ha='center')
    arrow([(849, 258), (849, 322)])
    text(874, 290, r'$\hat{\delta}_h$', 18, ha='center')
    rect(783, 322, 132, 48, '#EBF4F0', '#9BBFB0')
    text(849, 346, 'Denormalize', 14, ha='center')
    arrow([(915, 346), (972, 346)])
    text(944, 323, r'$\delta_h$', 18, ha='center')
    rect(972, 322, 100, 48, '#EBF4F0', '#9BBFB0')
    text(1022, 346, 'Integrate', 14, ha='center')
    arrow([(1022, 370), (1022, 392)])
    text(1022, 411, r'$\hat{s}_{h+1}$', 18, ha='center')

    # Alternatives, never summed: Updated carries c_h before readout transform.
    rect(275, 325, 245, 65, '#E9F3FA', '#85B9D5')
    text(397.5, 339, 'Frozen', 15, BLUE, 'bold', 'center')
    text(397.5, 371, r'$m_{h+1}=m_0$', 17, BLUE, ha='center')
    dot(226, 232, BLUE)
    arrow([(226, 232), (226, 354), (275, 354)], BLUE, 1.8)
    rect(553, 325, 206, 65, '#FFF0E8', '#E7AF8C')
    text(656, 339, 'Updated', 15, ORANGE, 'bold', 'center')
    text(656, 371, r'$m_{h+1}=c_h$', 17, ORANGE, ha='center')
    arrow([(642, 235), (642, 325)], ORANGE, 1.8)
    text(275, 411, r'Next step: $(\hat{s}_{h+1},m_{h+1})$; initially $\hat{s}_0=s_t$.', 15, MUTED)

    heading(465, 'B', 'Matched comparisons, separately trained models')
    # Primary comparison and auxiliary access control are distinct families.
    rect(14, 490, 490, 100, '#FBFCFD')
    rect(524, 490, 562, 100, '#FBFCFD')
    text(30, 510, 'Primary family · 39,497 / arm', 15, weight='bold')
    text(30, 542, 'Frozen / Updated × memory-input LN', 14.5)
    text(30, 573, r'$N_o=I$; placement control: $N_r,N_o\in\{I,\mathrm{LN}\}$', 14, MUTED)
    text(540, 510, 'Auxiliary context · 51,785 / arm', 15, weight='bold')
    text(540, 542, r'$c_h=\mathrm{GRUCell}([x_h,q_h],m_h)$; $m_{h+1}=c_h$', 15)
    text(540, 573, r'Fixed $q_h=m_0$ vs. dynamic $q_h=m_h$; no LN', 15, MUTED)

    heading(618, 'C', 'Three questions linking prediction to control')
    cards = [(14, '1  Observe the reversal'), (382, '2  Change future actions'),
             (750, '3  Vary data and budget')]
    for x, title in cards:
        rect(x, 644, 336, 151, '#FBFCFD')
        ax.plot([x+12, x+324], [681, 681], color='#DCE3E8', lw=1)
        text(x+14, 663, title, 15.5, weight='bold', region=(x+8, 644, x+328, 681))
    arrow([(354, 719), (378, 719)], LINE, 1.5, True)
    arrow([(722, 719), (746, 719)], LINE, 1.5, True)

    text(182, 701, 'Same trained models', 15, ha='center')
    arrow([(182, 711), (182, 720), (103, 720), (103, 726)])
    arrow([(182, 720), (261, 720), (261, 726)])
    rect(30, 726, 145, 41, '#EDF2F6')
    rect(189, 726, 145, 41, '#EDF2F6')
    text(102.5, 747, 'Logged\nprediction', 12.5, ha='center')
    text(261.5, 747, 'Closed-loop\ncontrol', 12.5, ha='center')
    text(182, 779, '+ physical-position audit', 14, MUTED, ha='center')

    text(550, 701, 'Fixed history, state, paired noise', 14, ha='center')
    rect(398, 716, 136, 40, '#EDF2F6')
    rect(566, 716, 136, 40, '#EDF2F6')
    text(466, 735, 'Logged\nactions', 12.5, ha='center')
    text(634, 735, 'Candidate\nactions', 12.5, ha='center')
    arrow([(466, 756), (466, 764), (550, 764), (550, 771)])
    ax.plot([634, 634, 550], [756, 764, 764], color=MUTED, lw=1.5)
    text(550, 784, 'Prediction error', 14, ha='center')

    text(918, 703, '4 recipes × 2 budgets × F / U', 15, ha='center')
    text(918, 730, 'Dense · Repeated', 13, MUTED, ha='center')
    text(918, 754, 'Re-simulated · Mixed', 13, MUTED, ha='center')
    text(918, 782, 'Planner queries + control', 13.5, ha='center')
    text(16, 820, 'Scope checks: fresh trajectories · planning and plant changes · ungated rollout decoder', 14, MUTED)

    # Enforce both canvas bounds and card-title bounds before any export.
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    for item in ax.texts:
        b = item.get_window_extent(renderer)
        if not fig.bbox.contains(b.x0, b.y0) or not fig.bbox.contains(b.x1, b.y1):
            raise ValueError(f'Text outside canvas: {item.get_text()!r}')
    for item, (x0, y0, x1, y1) in bounded_text:
        b = item.get_window_extent(renderer)
        lo = ax.transData.transform((x0, y1)); hi = ax.transData.transform((x1, y0))
        if b.x0 < lo[0] or b.x1 > hi[0] or b.y0 < lo[1] or b.y1 > hi[1]:
            raise ValueError(f'Text outside card: {item.get_text()!r}')
    for i, left in enumerate(ax.texts):
        a = left.get_window_extent(renderer)
        for right in ax.texts[i+1:]:
            b = right.get_window_extent(renderer)
            if min(a.x1,b.x1)-max(a.x0,b.x0)>1 and min(a.y1,b.y1)-max(a.y0,b.y0)>1:
                raise ValueError(f'Overlapping labels: {left.get_text()!r} / {right.get_text()!r}')
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(out/f'figure1_architecture.{ext}', dpi=180, facecolor='white')
    svg = out/'figure1_architecture.svg'
    svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__ == '__main__':
    draw()
