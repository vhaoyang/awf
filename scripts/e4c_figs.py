#!/usr/bin/env python
"""AWF E4C — E4 诊断图生成（图 + 图数据, 交付规范第2项）

运行号: AWF_E4C_FIGS_R1 (服务器端自编, 非本侧任务号)

输入: outputs/e4_diag/*.csv (E4 产出)
输出: outputs/e4_diag/figs/
  fig1_bias_terrain_season.png   地形x季节 偏差热图 (+ fig1_data.csv)
  fig2_bias_speedbin_lead.png    风速箱xlead 条件偏差热图 (+ fig2_data.csv)
  fig3_bias_by_lead.png          分lead 偏差/RMSE/MAE (+ fig3_data.csv)
  fig4_direrr_by_lead.png        风向圆周误差分lead (+ fig4_data.csv)
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("/home/vhyang/zcode/awf/outputs/e4_diag")
FIGS = OUT / "figs"
ORDER_SEASON = ["DJF", "MAM", "JJA", "SON"]


def heatmap(ax, mat, xlabels, ylabels, title, cmap="RdBu_r", vmin=None, vmax=None,
            unit="m/s"):
    v = np.nan_to_num(mat.values, nan=0.0)
    lim = vmax or np.nanmax(np.abs(mat.values)) or 1.0
    im = ax.imshow(mat.values, cmap=cmap, vmin=-(vmin or lim), vmax=(vmin or lim),
                   aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                        fontsize=8, color="black")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=f"bias ({unit})")


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    made = []

    # --- fig1: 地形 x 季节 ---
    ts = pd.read_csv(OUT / "bias_by_terrain_season.csv", index_col=0)
    ts = ts[[c for c in ORDER_SEASON if c in ts.columns]]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    heatmap(ax, ts, list(ts.columns), list(ts.index),
            "GFS wind-speed bias by terrain x season\n(forecast minus observation, station-hour mean)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_bias_terrain_season.png", dpi=150)
    ts.to_csv(FIGS / "fig1_data.csv")
    plt.close(fig); made.append("fig1")

    # --- fig2: 风速箱 x lead ---
    sb = pd.read_csv(OUT / "bias_by_speedbin.csv")
    mat = sb.pivot(index="obs_bin", columns="lead_h", values="bias")
    ns = sb.pivot(index="obs_bin", columns="lead_h", values="n")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    heatmap(ax, mat, [f"{c}h" for c in mat.columns], list(mat.index),
            "Conditional bias by observed-speed bin x lead time\n"
            "(positive = GFS over-forecast)")
    # n 标注在第二行小字
    for i, ob in enumerate(mat.index):
        for j, lh in enumerate(mat.columns):
            n = ns.loc[ob, lh] if ob in ns.index else np.nan
            if n == n:
                ax.text(j, i + 0.32, f"n={int(n)}", ha="center", va="center",
                        fontsize=5.5, color="gray")
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_bias_speedbin_lead.png", dpi=150)
    mat.round(3).to_csv(FIGS / "fig2_data.csv")
    plt.close(fig); made.append("fig2")

    # --- fig3: 分 lead ---
    bl = pd.read_csv(OUT / "bias_by_lead.csv")
    bl = bl.sort_values("lead_h")
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.errorbar(bl.lead_h, bl.bias, yerr=bl.rmse - bl.bias.abs(),
                fmt="o-", capsize=4, label="bias")
    ax.plot(bl.lead_h, bl.mae, "s--", ms=4, label="MAE")
    ax.plot(bl.lead_h, bl.rmse, "^:", ms=4, label="RMSE")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("lead time (h)"); ax.set_ylabel("m/s")
    ax.set_title("Wind-speed error by lead time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_bias_by_lead.png", dpi=150)
    bl.to_csv(FIGS / "fig3_data.csv", index=False)
    plt.close(fig); made.append("fig3")

    # --- fig4: 风向圆周误差 ---
    try:
        de = pd.read_csv(OUT / "dir_error_circ.csv").sort_values("lead_h")
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        ax.bar(de.lead_h.astype(str), de.circ_mean_deg,
               yerr=de.circ_mae, capsize=4, color="steelblue")
        ax.axhline(0, color="gray", lw=0.8)
        ax.axhline(10, color="crimson", lw=0.8, ls="--", label="B4 threshold 10deg")
        ax.set_xlabel("lead time (h)"); ax.set_ylabel("degrees")
        ax.set_title("Direction circular bias by lead (dir_trust only)\n"
                     "error bars: circular MAE")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGS / "fig4_direrr_by_lead.png", dpi=150)
        de.to_csv(FIGS / "fig4_data.csv", index=False)
        plt.close(fig); made.append("fig4")
    except FileNotFoundError:
        pass

    (FIGS / "figs_manifest.json").write_text(json.dumps(dict(
        run_id="AWF_E4C_FIGS_R1",
        utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        figures=made), ensure_ascii=False, indent=2))
    print("figures:", made, "->", FIGS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
