#!/usr/bin/env python
"""Plot ten-bin randomized PIT summaries from their saved diagnostic tables.

The first nine bins are left-closed and right-open; the last includes one.
Histogram shape is descriptive and does not by itself establish calibration."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
OUTD = "outputs/manuscript_ams"

rep = json.loads(open(f"{OUTD}/pit/pit_report.json").read())
counts = pd.read_csv(f"{OUTD}/pit/pit_counts.csv", index_col="bin")
# Use left-closed bins; the final bin includes one. Counts are unchanged.
counts.index = [f"[{i/10:.1f},{(i+1)/10:.1f})" if i < 9 else "[0.9,1.0]"
                for i in range(10)]
counts.to_csv(f"{OUTD}/pit/pit_counts.csv")
n = 18235

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
x = np.arange(10) * 0.1 + 0.05   # 箱中心(概率单位), 宽0.092
for ax, mdl in zip(axes, ("BQN", "QRF", "LGBM")):
    freq = counts[mdl].values / n
    ax.bar(x, freq, width=0.092, color="#7fa8d0", ec="#2c6faa", lw=0.5)
    ax.axhline(0.1, color="#888", lw=0.8, ls="--")
    ep = rep["PIT"][mdl]
    note = f"n = {n:,}"
    if ep.get("n_endpoint_1", 0):
        note += f"; {ep['n_endpoint_1']} at U = 1"
    if ep.get("n_endpoint_0", 0):
        note += f"; {ep['n_endpoint_0']} at U = 0"
    ax.text(0.97, 0.95, note, transform=ax.transAxes, ha="right",
            va="top", fontsize=6.2, color="#333")
    ax.set_title(mdl, fontsize=9)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("randomized PIT value (bin edges)", fontsize=7.5)
    ax.set_ylim(0, max(0.16, freq.max() * 1.18))
axes[0].set_ylabel("frequency", fontsize=7.5)
plt.tight_layout()
plt.savefig(f"{OUTD}/figs/fig_pit_T.png", dpi=300, bbox_inches="tight")
plt.close()
print("fig_pit_T done")
