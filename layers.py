"""Draws the layer diagram, data to model to prediction to decision, as fig/layers.png.

Usage: python layers.py

The wording follows PROPOSAL.md and the first decisions.md entry.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

LAYERS = [
    ("Data", "NYC TLC yellow-taxi trips\nMidtown -> JFK, Jan-Mar 2024\nmetered minutes; pickup\nhour, day of week, zone"),
    ("Model", "quantile regression at the\n90th percentile, against\na fixed allowance; fit on\nJanuary, offset on February"),
    ("Prediction", "minutes that 9 of 10\ncomparable rides stay under,\nfor the hour, day and zone\nof departure"),
    ("Decision", "allowance B to reserve;\nloss 0.9 x minutes short\n+ 0.1 x minutes unused,\nscored on March"),
]
BLUE, ORANGE = "#2f6fd0", "#e2672f"

fig, ax = plt.subplots(figsize=(14, 3.4))
ax.set_xlim(0, 14)
ax.set_ylim(0, 3.4)
ax.axis("off")

width, gap, left = 2.9, 0.6, 0.35
for i, (title, body) in enumerate(LAYERS):
    x = left + i * (width + gap)
    edge = ORANGE if title == "Decision" else BLUE
    face = "#fdeee7" if title == "Decision" else "#edf3fc"
    ax.add_patch(FancyBboxPatch((x, 0.3), width, 2.8, boxstyle="round,pad=0.02,rounding_size=0.12",
                                linewidth=2, edgecolor=edge, facecolor=face))
    ax.text(x + width / 2, 2.7, title, ha="center", va="center", fontsize=15, fontweight="bold")
    ax.text(x + width / 2, 1.45, body, ha="center", va="center", fontsize=10.5, color="#333333", linespacing=1.5)
    if i < len(LAYERS) - 1:
        ax.annotate("", xy=(x + width + gap - 0.08, 1.7), xytext=(x + width + 0.08, 1.7),
                    arrowprops={"arrowstyle": "-|>", "color": "#555555", "linewidth": 2})

Path("fig").mkdir(exist_ok=True)
fig.savefig("fig/layers.png", dpi=150, bbox_inches="tight", facecolor="white")
print("wrote fig/layers.png")
