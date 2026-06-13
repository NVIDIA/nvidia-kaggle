import json
from pathlib import Path
import matplotlib.pyplot as plt
p=Path(__file__).with_suffix('.json')
data=json.loads(p.read_text())
series=data['series']
labels=[s['label'] for s in series]
vals=[s['value'] for s in series]
fig, ax=plt.subplots(figsize=(8,4.5))
bars=ax.bar(labels, vals, color=['#4c78a8','#f58518','#54a24b','#b279a2'])
ax.set_title(data['title'])
ax.set_ylabel('Value')
ax.tick_params(axis='x', rotation=20)
ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
fig.tight_layout(); fig.savefig(Path(__file__).with_suffix('.png'), dpi=160)
