import json
from pathlib import Path
import matplotlib.pyplot as plt
p=Path(__file__).with_suffix('.json')
data=json.loads(p.read_text())
labels=[s['label'] for s in data['series']]
vals=[s['value'] for s in data['series']]
centers=[(s['bin_left']+s['bin_right'])/2 for s in data['series']]
widths=[s['bin_right']-s['bin_left'] for s in data['series']]
fig, ax=plt.subplots(figsize=(9,5))
ax.bar(centers, vals, width=widths, color='#4c78a8', alpha=.85, align='center')
ax.axvline(data['median_rows'], color='#f58518', lw=2, label=f"median={data['median_rows']:.0f}")
ax.set_title(data['title'])
ax.set_xlabel('Rows per horizontal well')
ax.set_ylabel('Number of wells')
ax.legend()
fig.tight_layout(); fig.savefig(Path(__file__).with_suffix('.png'), dpi=160)
