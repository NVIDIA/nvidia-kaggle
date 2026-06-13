import json, pathlib, textwrap
import matplotlib.pyplot as plt
js=pathlib.Path(__file__).with_suffix('.json')
data=json.loads(js.read_text())
series=data['series']
labels=[s['label'] for s in series]
values=[s['value'] for s in series]
n=len(series)
fig, ax=plt.subplots(figsize=(10, max(4, 0.35*n+1.5)))
y=list(range(n))
ax.barh(y, values, color='#76b900')
ax.set_yticks(y)
ax.set_yticklabels([textwrap.shorten(l, width=38, placeholder='…') for l in labels], fontsize=8)
ax.invert_yaxis()
ax.set_title(data['title'], fontsize=11)
ax.grid(axis='x', alpha=0.25)
for i,v in enumerate(values):
    ax.text(v, i, f' {v:g}', va='center', fontsize=8)
ax.set_xlabel('RMSE / score (lower is better)' if ('RMSE' in data['title'] or 'scores' in data['title']) else 'count')
fig.tight_layout()
fig.savefig(js.with_suffix('.png'), dpi=180)
