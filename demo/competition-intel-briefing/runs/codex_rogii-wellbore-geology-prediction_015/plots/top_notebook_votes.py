
import json
from pathlib import Path
import matplotlib.pyplot as plt

json_path = Path(__file__).with_suffix('.json')
out_path = Path(__file__).with_suffix('.png')
data = json.load(open(json_path))
series = data['series']
labels = [item['label'] for item in series]
values = [item['value'] for item in series]
height = max(3.8, 0.38 * len(series) + 1.4)
fig, ax = plt.subplots(figsize=(11, height))
y = range(len(series))
ax.barh(list(y), values, color='#2a6fbb')
ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_title(data['title'], loc='left', fontsize=12, pad=12)
ax.set_xlabel('Value')
ax.grid(axis='x', alpha=0.25)
for idx, value in enumerate(values):
    ax.text(value, idx, f' {value:g}', va='center', fontsize=8)
fig.tight_layout()
fig.savefig(out_path, dpi=180, bbox_inches='tight')
