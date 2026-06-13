import json
from pathlib import Path
import matplotlib.pyplot as plt
p=Path(__file__).with_suffix('.json')
data=json.loads(p.read_text())
series=data['series']
labels=[s['label'] for s in series]
vals=[s['value'] for s in series]
fig, ax=plt.subplots(figsize=(9,7))
ax.barh(labels, vals, color='#54a24b')
ax.set_title(data['title'])
ax.set_xlabel('TVT max - min (ft)')
ax.set_ylabel('Well')
fig.tight_layout(); fig.savefig(Path(__file__).with_suffix('.png'), dpi=160)
