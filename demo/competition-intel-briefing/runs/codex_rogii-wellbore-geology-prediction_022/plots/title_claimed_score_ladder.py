import json
from pathlib import Path
import matplotlib.pyplot as plt
p=Path(__file__).with_suffix('.json')
data=json.loads(p.read_text())
series=sorted(data['series'], key=lambda s: (s['provenance'], s['value']))
labels=[s['label'] for s in series][::-1]
vals=[s['value'] for s in series][::-1]
fig, ax=plt.subplots(figsize=(10,6))
bars=ax.barh(labels, vals, color='#e45756', hatch='///', edgecolor='#7f1d1d', label='title-claim, unverified')
ax.set_title(data['title'])
ax.set_xlabel('Claimed RMSE/LB/CV value; lower is better')
if vals:
    ax.set_xlim(min(vals)-0.5, max(vals)+0.8)
ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
ax.legend(loc='lower right')
fig.tight_layout(); fig.savefig(Path(__file__).with_suffix('.png'), dpi=160)
