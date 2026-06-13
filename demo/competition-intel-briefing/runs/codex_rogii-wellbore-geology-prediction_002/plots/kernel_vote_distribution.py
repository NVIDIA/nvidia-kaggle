import json, pathlib
import matplotlib.pyplot as plt
here = pathlib.Path(__file__).resolve().parent
name = pathlib.Path(__file__).stem
payload = json.loads((here / f"{name}.json").read_text())
series = payload["series"]
labels = [item["label"] for item in series]
values = [float(item["value"]) for item in series]
height = max(4, 0.35 * len(labels) + 1.5)
fig, ax = plt.subplots(figsize=(10, height))
y = range(len(labels))
ax.barh(list(y), values, color="#4C78A8")
ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel(payload.get("xlabel", "Value"))
ax.set_title(payload["title"])
for idx, value in enumerate(values):
    ax.text(value, idx, f" {value:g}", va="center", fontsize=8)
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig(here / f"{name}.png", dpi=160)
