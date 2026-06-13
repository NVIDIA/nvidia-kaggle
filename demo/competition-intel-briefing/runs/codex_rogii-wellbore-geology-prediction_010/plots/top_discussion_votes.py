import json
from pathlib import Path
import matplotlib.pyplot as plt
import sys
path = Path(sys.argv[1])
with open(path) as f:
    obj = json.load(f)
series = obj['series']
labels = [s['label'] for s in series]
values = [s['value'] for s in series]
height = max(4, 0.36 * len(series) + 1.3)
fig, ax = plt.subplots(figsize=(11, height))
y = list(range(len(series)))
ax.barh(y, values, color='#4C78A8')
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_title(obj['title'], fontsize=12, pad=12)
ax.set_xlabel('Value')
ax.grid(axis='x', alpha=0.25)
for i, v in enumerate(values):
    ax.text(v, i, f' {v:.0f}' if abs(v) >= 10 else f' {v:.2f}', va='center', fontsize=8)
fig.tight_layout()
fig.savefig(path.with_suffix('.png'), dpi=160, bbox_inches='tight')
