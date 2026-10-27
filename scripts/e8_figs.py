#!/usr/bin/env python
"""AWF E8 — 图4 重生成 (V3-07: 数字与统一结果表同源)

运行号: AWF_E8_FIGS_R1
面板: (a) MAE; (b) QS3 + 完整CRPS(单一完整分布, 精确积分);
      (c) POD–POFD 阈值扫描 + 匹配 POFD 0.02/0.05/0.10 标记。
数据源: outputs/review_v3/{scores_summary_v3.json, threshold_curves_v3.csv}
输出: outputs/manuscript_v4/figs/fig4_methods.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

V3 = Path("/home/vhyang/zcode/awf/outputs/review_v3")
OUT = Path("/home/vhyang/zcode/awf/outputs/manuscript_v5/figs")
s = json.load(open(V3 / "scores_summary_v3.json"))["scores"]

mae_map = {"GFS": 1.975, "QM": 1.901, "MOS": 1.796, "EMOS": 1.802,
           "stnMOS": s["MAE_stnMOS"], "LGBM": s["MAE_q50_mono"]}
fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.1))

names = list(mae_map)
axes[0].bar(names, [mae_map[n] for n in names], color="#0173B2")
axes[0].set_ylabel("MAE (m/s)")
axes[0].set_title("(a) point skill (MAE)")
axes[0].tick_params(axis="x", rotation=45)
for lbl in axes[0].get_xticklabels():
    lbl.set_ha("right")
axes[0].set_ylim(1.6, 2.1)
axes[0].axhline(s["MAE_q50_raw"], color="#D55E00", lw=0.8, ls=":",
                label="LGBM raw median")
axes[0].legend(fontsize=6.5)

qs = {"EMOS": (s["QS3_EMOS"], s["CRPS_EMOS"]),
      "EMOS-geo": (np.nan, s["CRPS_EMOSgeo_refit"]),
      "LGBM": (s["QS3_LGBM"], s["CRPS_exact_LGBM"])}
x2 = np.arange(len(qs))
axes[1].bar(x2 - 0.18, [qs[k][0] for k in qs], width=0.34,
            color="#029E73", label="QS$_3$")
axes[1].bar(x2 + 0.18, [qs[k][1] for k in qs], width=0.34,
            color="#D55E00", label="CRPS (full dist., exact)")
axes[1].set_xticks(x2, list(qs))
axes[1].set_ylim(1.0, 1.45)
axes[1].set_title("(b) distributional scores")
axes[1].legend(fontsize=6.5)

tc = pd.read_csv(V3 / "threshold_curves_v3.csv")
colors = {"GFS": "0.4", "LGBM_q90": "#0173B2", "EMOS_q90": "#D55E00"}
marks = {"GFS": "s", "LGBM_q90": "o", "EMOS_q90": "^"}
# V5-01: cap 操作点 = POFD<=cap 内最高 POD(与 e9_derived.py 同规则),
# 全精度坐标记录; 画点时做"从右向左压缩"显示错开: 每 cap 内按 POFD 降序
# 排列并保持 >=0.0045 的水平间距(仅向左移, 永不超过 cap) —— 错开仅为
# 显示, CSV/JSON 保留全精度真实坐标
marker_rows = []
ops = {cap: {} for cap in (0.02, 0.05, 0.10)}
for name, g in tc.groupby("method"):
    axes[2].plot(g.POFD, g.POD, color=colors.get(name, "k"), lw=1.2,
                 label=name.replace("_q90", ""))
    for cap in (0.02, 0.05, 0.10):
        gg = g[g.POFD <= cap]
        if len(gg) == 0:
            continue
        r = gg.loc[gg.POD.idxmax()]
        ops[cap][name] = r
        marker_rows.append(dict(method=name, cap=cap,
                                threshold=float(r.threshold),
                                POFD=float(r.POFD), POD=float(r.POD)))
MIN_SEP = 0.0045
for cap, d in ops.items():
    order = sorted(d.items(), key=lambda kv: -kv[1].POFD)
    prev_x = None
    for name, r in order:
        x = float(r.POFD)
        if prev_x is not None:
            x = min(x, prev_x - MIN_SEP)
        x = max(x, 0.0)
        axes[2].plot(x, float(r.POD), marks[name], ms=6,
                     color=colors[name], mec="white", mew=0.6, zorder=5,
                     linestyle="none")
        prev_x = x
pd.DataFrame(marker_rows).to_csv(
    OUT.parent / "fig4_markers_v5.csv", index=False)
axes[2].set_xlabel("POFD (F/(F+C))")
axes[2].set_ylabel("POD (H/(H+M))")
axes[2].set_title("(c) strong-wind threshold sweep "
                    "(POFD $\\leq$ 0.12)", fontsize=9)
axes[2].set_xlim(0.0, 0.12)
leg = axes[2].legend(fontsize=6.5, loc="lower right",
                     title="obs>15 m/s; markers: POFD caps .02/.05/.10")
leg.get_title().set_fontsize(5.5)
leg.get_title().set_color("0.25")

fig.tight_layout(w_pad=2.2)
fig.savefig(OUT / "fig4_methods.png", bbox_inches="tight", dpi=200)
print("fig4 saved:", OUT / "fig4_methods.png")
