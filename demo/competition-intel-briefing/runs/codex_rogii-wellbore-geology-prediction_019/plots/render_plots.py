import json, pathlib, textwrap
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plot_dir=pathlib.Path(__file__).resolve().parent
files=[
    'public_leaderboard_top20_rmse',
    'score_ladder_verified_and_title_claims',
    'public_notebooks_top20_votes',
    'discussions_top15_comments',
]
for stem in files:
    data=json.loads((plot_dir/f'{stem}.json').read_text())
    series=data['series']
    labels=[s['label'] for s in series]
    values=[s['value'] for s in series]
    prov=[s['provenance'] for s in series]
    height=max(5, 0.38*len(series)+1.6)
    fig, ax=plt.subplots(figsize=(11, height))
    ypos=list(range(len(series)))
    colors=[]; hatches=[]
    for p in prov:
        if p == 'verified':
            colors.append('#1f77b4'); hatches.append('')
        elif p == 'title-claim':
            colors.append('#ffbb78'); hatches.append('///')
        else:
            colors.append('#2ca02c'); hatches.append('xx')
    bars=ax.barh(ypos, values, color=colors, edgecolor='black', linewidth=0.4)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_yticks(ypos)
    ax.set_yticklabels([textwrap.fill(l, 42) for l in labels], fontsize=8)
    ax.invert_yaxis()
    if 'rmse' in stem or 'score_ladder' in stem:
        ax.set_xlabel('RMSE / score (lower is better)')
    elif 'votes' in stem:
        ax.set_xlabel('Votes')
    else:
        ax.set_xlabel('Comments')
    ax.set_title(data['title'], fontsize=11)
    ax.grid(axis='x', alpha=0.25)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width(), bar.get_y()+bar.get_height()/2, f' {v:g}', va='center', fontsize=8)
    legend=[]
    if 'verified' in prov:
        legend.append(Patch(facecolor='#1f77b4', edgecolor='black', label='verified from this run'))
    if 'title-claim' in prov:
        legend.append(Patch(facecolor='#ffbb78', edgecolor='black', hatch='///', label='author title-claim, unverified'))
    if 'derived' in prov:
        legend.append(Patch(facecolor='#2ca02c', edgecolor='black', hatch='xx', label='derived'))
    if legend:
        ax.legend(handles=legend, loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir/f'{stem}.png', dpi=160)
    plt.close(fig)
print('rendered', ', '.join(files))
