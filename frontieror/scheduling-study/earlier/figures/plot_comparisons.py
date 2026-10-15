"""Reproduce matched endpoint figures from the accompanying public-results.json.
Run with Python and matplotlib: python plot_comparisons.py
This script reads the downloaded snapshot only and never runs a solver.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
rows = json.loads((root / 'public-results.json').read_text())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
for budget in [300, 1200]:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), layout='constrained', sharex=True)
    for seed, ax in enumerate(axes):
        for index in range(5):
            case = f'large-{index+1}'
            pick = lambda alg: next(r for r in rows if r['algorithm'] == alg and r['budget_s'] == budget and r['seed'] == seed and r['instance'] == case)
            a, b = pick('v2'), pick('LNS')
            assert a['input_sha256'] == b['input_sha256']
            ax.plot([a['objective'], b['objective']], [index, index], color='#BAC2C9', lw=2, zorder=1)
            ax.scatter(a['objective'], index, color='#0072B2', s=44, label='v2' if index == 0 else None, zorder=3)
            ax.scatter(b['objective'], index, color='#D55E00', s=36, marker='s', label='CP-SAT LNS' if index == 0 else None, zorder=3)
        ax.set(yticks=range(5), yticklabels=[f'large-{i}' for i in range(1,6)], xlabel='Total weighted tardiness', title=f'Seed {seed} | {budget:,} s requested')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=.2)
        ax.legend()
        ax.margins(x=.12,y=.2)
    for ext in ['svg','pdf','png']:
        fig.savefig(root / f'reproduced_{budget}s.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
