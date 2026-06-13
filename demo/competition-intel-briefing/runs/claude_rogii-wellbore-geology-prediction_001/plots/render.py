"""Render each plot PNG from its sidecar JSON ({title, source, series}).

Every bar's height is the JSON `value` (a gathered number), so the image can
never disagree with the saved data. Bars are styled by `provenance`:
  verified   -> solid
  title-claim-> hatched (author's title number, unverified)
  derived    -> light/edged (computed aggregate, e.g. LB band cutoff)
Bars are sorted by value within each provenance group.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = Path(__file__).parent

STYLE = {
    "verified":    dict(color="#2c6fbb", hatch=None),
    "title-claim": dict(color="#c0c0c0", hatch="///"),
    "derived":     dict(color="#7fb069", hatch=".."),
}


def render(stem):
    data = json.loads((HERE / f"{stem}.json").read_text())
    series = list(data["series"])
    # sort by value within provenance group, groups ordered verified/title/derived
    order = {"verified": 0, "title-claim": 1, "derived": 2}
    series.sort(key=lambda s: (order.get(s.get("provenance"), 9), s["value"]))

    labels = [s["label"] for s in series]
    values = [s["value"] for s in series]
    provs = [s.get("provenance", "verified") for s in series]

    fig, ax = plt.subplots(figsize=(10, max(3.2, 0.5 * len(series) + 1.4)))
    y = range(len(series))
    for i, (v, p) in enumerate(zip(values, provs)):
        st = STYLE.get(p, STYLE["verified"])
        ax.barh(i, v, color=st["color"], hatch=st["hatch"], edgecolor="#333", linewidth=0.6)
        ax.text(v, i, f" {v:g}", va="center", ha="left", fontsize=9)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()  # best (lowest value) on top for RMSE plots
    ax.set_xlabel("RMSE (lower is better)" if "RMSE" in data["title"] else "value")
    ax.set_title(data["title"], fontsize=11, fontweight="bold")
    ax.margins(x=0.14)
    ax.grid(axis="x", alpha=0.3)

    present = [p for p in ("verified", "title-claim", "derived") if p in provs]
    legend_txt = {
        "verified": "verified (gathered public score / LB)",
        "title-claim": "title-claim (author's number, unverified)",
        "derived": "derived (computed LB band cutoff)",
    }
    handles = [mpatches.Patch(facecolor=STYLE[p]["color"], hatch=STYLE[p]["hatch"],
                              edgecolor="#333", label=legend_txt[p]) for p in present]
    if len(present) > 1 or present == ["derived"]:
        ax.legend(handles=handles, fontsize=8, loc="lower right")

    fig.tight_layout()
    out = HERE / f"{stem}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    stems = sys.argv[1:] or [
        "public_notebook_ladder",
        "leaderboard_frontier",
        "discussion_engagement",
    ]
    for s in stems:
        render(s)
