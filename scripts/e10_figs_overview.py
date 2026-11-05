#!/usr/bin/env python
"""Draw the experiment framework and station-level method-comparison maps.

The framework distinguishes the three evaluation splits, two feature sets,
method-specific seed mixtures and the EMOS reference. The maps require
precomputed winner tables with station and winner columns. They do not
derive rankings from raw predictions; see docs/REPRODUCE.md for input paths."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os
from pathlib import Path

import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
OUT = "outputs/manuscript_ams/figs"
Path(OUT).mkdir(parents=True, exist_ok=True)
Path("outputs/manuscript_ams/pit").mkdir(parents=True, exist_ok=True)

# ============ 框架图 ============
fig, ax = plt.subplots(figsize=(7.2, 5.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 76)
ax.axis("off")


def box(x, y, w, h, text, fc="#eaf2fb", ec="#2c6faa", fs=7.2, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.35)


def arrow(x1, y1, x2, y2, style="-|>", lw=1.0, ls="-", color="#444444"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=11, lw=lw, ls=ls,
                                 color=color))


# ---- 数据层 (确定性输入) ----
box(2, 68, 44, 6.6, "GFS v16 0.25° deterministic (00Z, leads 0–72 h)\n50 pairing stations  (deterministic input)",
    fc="#fdf3e3", ec="#b8860b", fs=6.8)
box(54, 68, 44, 6.6, "Observations 2021 (AntAWS v2, QC-screened)\n57,700 valid rows  (deterministic input)",
    fc="#fdf3e3", ec="#b8860b", fs=6.8)
ax.text(50, 66.0, "paired by (station, valid time)", ha="center",
        fontsize=6.4, style="italic", color="#666")

# ---- 设计层: 3 划分 × 2 特征集 = 六设计矩阵 ----
ax.add_patch(FancyBboxPatch((2, 44.5), 96, 19, boxstyle="round,pad=0.6",
                            fc="#fcfcfc", ec="#999", lw=0.9))
ax.text(4.5, 61.6, "Six frozen designs  =  3 split schemes × 2 feature sets  (frozen before any scoring)",
        fontsize=7.4, weight="bold")
ax.text(46, 58.4, "F0 — 13 base features (incl. lat/lon)", ha="center",
        fontsize=6.6, weight="bold", color="#7a4aa0")
ax.text(80, 58.9, "F1 = F0 + 8 REMA terrain", ha="center",
        fontsize=6.6, weight="bold", color="#7a4aa0")
ax.text(80, 57.3, "(appended, not replaced)", ha="center",
        fontsize=6.0, color="#7a4aa0")
rows = [("T — temporal split (by valid time)\nMar–Aug train | Sep select | Oct–Dec test", 54.0),
        ("S — leave-one-out stations\n50 station folds, all months", 50.6),
        ("F — forward stations\ntrain ≤ 31 Aug → test held-out stn ≥ 1 Oct", 47.2)]
for lbl, yc in rows:
    ax.text(27, yc, lbl, ha="right", va="center", fontsize=5.8, linespacing=1.3)
chips = [("T", 46, 54.0), ("S", 46, 50.6), ("F", 46, 47.2),
         ("T1", 80, 54.0), ("S1", 80, 50.6), ("F1", 80, 47.2)]
for name, xc, yc in chips:
    box(xc - 12, yc - 1.5, 24, 3.0, name, fc="#f3eafb", ec="#7a4aa0", fs=7.6)

# ---- 方法层: 三模型 + EMOS 参考 ----
box(2, 28, 24, 12, "BQN\nMLP → softplus\ncumulative increments →\ndegree-D Bernstein Q(τ)", fc="#e8f6ec", ec="#2e8b57", fs=6.6)
box(29, 28, 24, 12, "QRF\n500 trees → native\nweighted leaf empirical\nCDFs", fc="#e8f6ec", ec="#2e8b57", fs=6.6)
box(56, 28, 24, 12, "LightGBM\n22 quantile boosters →\nrearrangement →\npiecewise-linear Q(τ)", fc="#e8f6ec", ec="#2e8b57", fs=6.6)
ax.add_patch(FancyBboxPatch((84, 28), 14, 12, boxstyle="round,pad=0.6",
                            fc="#fdecec", ec="#c0392b", lw=1.0, ls=(0, (4, 2))))
ax.text(91, 34, "EMOS\nreference\n(temporal\ndesign only)", ha="center", va="center",
        fontsize=6.6, linespacing=1.35, color="#7a2000")

# ---- 种子混合层 (仅三模型主路径) ----
box(2, 16, 78, 8.5, "Per method: 3-seed CDF mixture (seeds 17, 42, 2026, equal weight)\n"
    "every score is computed from the mixture CDF — never an average of member scores",
    fc="#eaf2fb", ec="#2c6faa", fs=6.8)

# ---- 验证层 (概率输出) ----
box(2, 3, 96, 10, "Verification  (probabilistic output)\n"
    "exact-decomposition CRPS (adaptive scoring for extreme-support folds) · MAE/bias · cov80/width80 · Brier(>15 m s⁻¹\n"
    "paired date-block bootstrap for cross-model differences · pooled + station-equal weightings",
    fc="#eef7f0", ec="#2e8b57", fs=6.8)

# ---- 箭头 ----
arrow(20, 68, 20, 63.8); arrow(70, 68, 70, 63.8)          # 数据 -> 设计
arrow(14, 44.5, 14, 40.8); arrow(41, 44.5, 41, 40.8)       # 设计 -> BQN/QRF/LGBM
arrow(68, 44.5, 68, 40.8)
arrow(91, 44.5, 91, 40.8, ls=(0, (4, 2)), color="#c0392b")  # 设计 -> EMOS (虚线)
arrow(14, 28, 14, 25.1); arrow(41, 28, 41, 25.1); arrow(68, 28, 68, 25.1)  # 模型 -> 混合
arrow(91, 28, 91, 13.8, ls=(0, (4, 2)), color="#c0392b")    # EMOS 旁路直下验证 (虚线)
arrow(41, 16, 41, 13.8)                                     # 混合 -> 验证
ax.text(50, 1.0, "Test rows never enter preprocessing, imputation, or model/epoch selection; "
        "outputs are marginal wind-speed distributions.",
        ha="center", fontsize=6.2, style="italic", color="#666")
plt.tight_layout()
plt.savefig(f"{OUT}/fig_framework.png", dpi=300, bbox_inches="tight")
plt.close()
print("fig_framework done")

# ============ 模型表现空间图 ============
reg = pd.read_csv(Path(os.environ.get("AWF_DATA_ROOT", "data"))
              / "derived/station_qc/station_sensor_registry.csv")
frames = []
for d in ["S", "S1", "F", "F1"]:
    g = pd.read_csv(Path(os.environ.get("AWF_WINNER_ROOT", "outputs/manuscript_ams/pit")) / f"winner_{d}.csv")
    g["design"] = d
    frames.append(g)
w = pd.concat(frames, ignore_index=True).merge(
    reg[["station", "lat", "lon"]], on="station", how="left")
w.to_csv("outputs/manuscript_ams/pit/"
         "fig_winner_data.csv", index=False)  # Data underlying the station-comparison map

CMAP = {"bqn": "#d95f02", "qrf": "#1b9e77", "lgbm": "#7570b3"}
fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.4),
                         subplot_kw={"projection": None})
titles = {"S": "S — leave-one-out (50 stns)", "S1": "S1 — LOO + terrain (50 stns)",
          "F": "F — forward (42 stns)", "F1": "F1 — forward + terrain (42 stns)"}
for ax, d in zip(axes.ravel(), ["S", "S1", "F", "F1"]):
    g = w[w.design == d]
    for m in ("bqn", "qrf", "lgbm"):
        gg = g[g.winner == m]
        ax.scatter(gg.lon, gg.lat, s=16, c=CMAP[m], label=m.upper(),
                   edgecolors="white", linewidths=0.3, zorder=3)
    if d in ("S1", "F1"):
        nico = g[g.station == "Nico"]
        if len(nico):
            ax.scatter(nico.lon, nico.lat, s=110, facecolors="none",
                       edgecolors="#d95f02", linewidths=1.8, zorder=4)
            nlon = nico.lon.values[0]
            left = nlon > 0   # 东经站标注放左侧, 防右缘裁剪
            ax.annotate("Nico (BQN anomaly)",
                        (nlon, nico.lat.values[0]),
                        textcoords="offset points",
                        xytext=(-10, 12) if left else (10, 12),
                        ha="right" if left else "left",
                        va="bottom",
                        fontsize=6.0, color="#a04000")
    tally = g.winner.value_counts()
    ax.text(0.02, 0.03,
            "  ".join(f"{m.upper()} {tally.get(m, 0)}" for m in ("bqn", "qrf", "lgbm")),
            transform=ax.transAxes, fontsize=6.0, color="#333",
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))
    ax.set_title(titles[d], fontsize=8)
    ax.set_xlim(-185, 185)
    ax.set_ylim(-90, -60)
    ax.set_xticks([-150, -90, -30, 30, 90, 150])
    ax.set_yticks([-85, -75, -65])
    ax.grid(alpha=0.25, lw=0.4)
    ax.axhline(-66.5, color="#9aa", lw=0.6, ls="--")
    ax.text(-182, -66.0, "Antarctic Circle", fontsize=5.5, color="#777")
axes[0, 0].set_ylabel("latitude (°)")
axes[1, 0].set_ylabel("latitude (°)")
axes[1, 0].set_xlabel("longitude (°)")
axes[1, 1].set_xlabel("longitude (°)")
handles = [plt.Line2D([], [], marker="o", ls="", color=CMAP[m],
                      label=f"{m.upper()} lowest") for m in CMAP]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.2,
           frameon=False, bbox_to_anchor=(0.5, 0.005))
fig.suptitle("Lowest observed mean per-station CRPS by station "
             "(descriptive; full network, all stations retained)",
             fontsize=8.4)
plt.tight_layout(rect=[0, 0.05, 1, 0.965])
plt.savefig(f"{OUT}/fig_winner_map.png", dpi=300, bbox_inches="tight")
plt.close()
print("fig_winner_map done")
