
from pathlib import Path
import json
import matplotlib.pyplot as plt
import sys
path = Path(sys.argv[1])
data = json.loads(path.read_text())
series = data['series']
labels = [s['label'] for s in series]
values = [s['value'] for s in series]
height = max(4.5, 0.38 * len(series) + 1.6)
fig, ax = plt.subplots(figsize=(10, height))
y = range(len(series))
ax.barh(list(y), values, color='#76B900')
ax.set_yticks(list(y), labels)
ax.invert_yaxis()
ax.set_title(data['title'])
ax.set_xlabel('Value')
ax.grid(axis='x', alpha=0.25)
for idx, value in enumerate(values):
    ax.text(value, idx, f' {value:g}', va='center', fontsize=9)
fig.tight_layout()
fig.savefig(path.with_suffix('.png'), dpi=180)
