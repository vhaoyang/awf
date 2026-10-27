#!/usr/bin/env python
"""AWF E6 — 论文图统一出图（WAF 投稿规格: 300dpi, 色盲安全, 6 图）

运行号: AWF_E6_FIGS_R1 (服务器端自编, 非本侧任务号)

图1 站点地图(南极极射赤面投影, 地形着色/机构形状) + 数据覆盖面板
图2 偏差条件结构: (a) 风速箱xlead 热图 (b) 地形x季节 热图
图3 分lead 偏差 + 高度订正对照(z0 带宽)
图4 方法对比: (a) MAE/CRPS (b) 分位可靠度 (c) 尾部
图5 LOO 泛化地图(逐站增益%)
图6 特征消融
投影/岸线: pyproj + 本地 cst10_polygon.shp (不依赖 cartopy 联网数据)
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
import shapefile
from matplotlib.collections import LineCollection
from pyproj import CRS, Transformer

DATA = Path("/mnt/d/data/awfdata")
OUT = Path("/home/vhyang/zcode/awf/outputs/e6_figs")
E4 = Path("/home/vhyang/zcode/awf/outputs/e4_diag")
E5 = Path("/home/vhyang/zcode/awf/outputs/e5_baseline")

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "figure.dpi": 150, "savefig.dpi": 300, "font.family": "DejaVu Sans",
})
TERR_COLORS = {"coast<100": "#0173B2", "100-500": "#DE8F05",
               "500-1000": "#029E73", "slope1-2k": "#D55E00",
               "plateau>2k": "#CC78BC"}
INSTIT_MARKER = {"AMRC": "o", "POLENET": "s", "CHINARE": "^",
                 "IMAU": "D", "PNRA": "v"}

CRS_SPS = CRS.from_proj4(
    "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +datum=WGS84")
TR = Transformer.from_crs(CRS.from_epsg(4326), CRS_SPS,
                          always_xy=True)


def coast_lines():
    """cst10_polygon.shp 坐标本就是 WGS84 南极极射赤面(标准纬线-71)米制,
    与显示投影一致, 直接返回不转换。"""
    r = shapefile.Reader(str(DATA / "raw/antaws/shapefiles/Shp/cst10_polygon"))
    segs = []
    for shp in r.shapes():
        pts = np.array([(p[0], p[1]) for p in shp.points])
        parts_idx = list(shp.parts) + [len(pts)]
        for a, b in zip(parts_idx[:-1], parts_idx[1:]):
            seg = pts[a:b]
            # 再按大跳变切割(防个别未分段的长直线)
            if len(seg) > 1:
                d = np.hypot(*(np.diff(seg, axis=0).T))
                brk = np.where(d > 100e3)[0]
                st = 0
                for x in list(brk) + [len(seg) - 1]:
                    if x > st:
                        segs.append(seg[st:x + 1])
                    st = x + 1
    return segs


def draw_map(ax, segs, title="", extent=3300e3):
    ax.set_facecolor("#E8F4F8")
    for seg in segs:
        ax.plot(seg[:, 0], seg[:, 1], color="0.30", lw=0.6)
    th = np.linspace(0, 2 * np.pi, 361)
    for rad in (1000e3, 2000e3, 3000e3):
        ax.plot(rad * np.cos(th), rad * np.sin(th), color="0.7",
                lw=0.4, ls=":")
    ax.set_xlim(-extent, extent); ax.set_ylim(-extent, extent)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for lbl, lon in [("60°W", -60), ("60°E", 60), ("150°W", -150),
                     ("120°E", 120)]:
        x, y = TR.transform(lon, -66)
        ax.text(x, y, lbl, fontsize=7, color="0.3", ha="center")
    if title:
        ax.set_title(title)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    reg = pd.read_csv(DATA / "derived/station_qc/station_sensor_registry.csv")
    loo = pd.read_csv(E5 / "loo/loo_by_station.csv")
    made = []

    # ---------------- 图1 站点地图 + 覆盖 ----------------
    segs = coast_lines()
    fig = plt.figure(figsize=(7.4, 4.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], hspace=0.42,
                          wspace=0.28, left=0.02, right=0.97,
                          top=0.90, bottom=0.15)
    ax = fig.add_subplot(gs[:, 0])
    draw_map(ax, segs, "(a) 50 pairing stations", extent=3450e3)
    paired = reg[reg.station.isin(loo.station)]
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", markersize=5,
                      color=TERR_COLORS[t], label=t) for t in TERR_COLORS]
    handles += [Line2D([], [], marker=INSTIT_MARKER[i], ls="", markersize=5,
                       markerfacecolor="none", markeredgecolor="k",
                       label=i) for i in INSTIT_MARKER]
    for terr, g in paired.groupby("terrain"):
        for inst, gg in g.groupby("institution"):
            x, y = TR.transform(gg.lon.values, gg.lat.values)
            ax.scatter(x, y, s=22, color=TERR_COLORS[terr],
                       marker=INSTIT_MARKER.get(inst, "o"),
                       edgecolor="k", lw=0.3)
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.30, 0.005), ncol=5, fontsize=6,
               frameon=True, framealpha=1.0, handletextpad=0.2,
               columnspacing=0.8,
               title="color: terrain  |  marker: institution",
               title_fontsize=6)
    ax2 = fig.add_subplot(gs[0, 1])
    days = pd.read_csv(DATA / "meta/gfs_2021_available_days.txt",
                       header=None, names=["d"], dtype=str)
    mcount = days.d.str[4:6].astype(int).value_counts().sort_index()
    ax2.bar(mcount.index, mcount.values, color="#0173B2")
    ax2.set_xlabel("month (2021)"); ax2.set_ylabel("GFS days available")
    ax2.set_title("(b) GFS archive coverage", fontsize=8.5)
    ax3 = fig.add_subplot(gs[1, 1])
    pair = pd.read_parquet(DATA / "derived/paired/paired_2021.parquet",
                           columns=["valid_time", "station"])
    pm = pair.valid_time.dt.month.value_counts().sort_index()
    ax3.bar(pm.index, pm.values, color="#D55E00")
    ax3.set_xlabel("month (2021)"); ax3.set_ylabel("paired rows")
    ax3.set_title("(c) paired obs–forecast rows", fontsize=8.5)
    fig.savefig(OUT / "fig1_stations_coverage.png")
    plt.close(fig); made.append("fig1")

    # ---------------- 图2 偏差条件结构 ----------------
    sb = pd.read_csv(E4 / "bias_by_speedbin.csv")
    mat = sb.pivot(index="obs_bin", columns="lead_h", values="bias")
    mat = mat.reindex(["0-3", "3-6", "6-10", "10-15", "15-25", ">25"])
    ts = pd.read_csv(E4 / "bias_by_terrain_season.csv", index_col=0)
    ts = ts[[c for c in ["DJF", "MAM", "JJA", "SON"] if c in ts.columns]]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    lim = np.nanmax(np.abs(mat.values))
    im = axes[0].imshow(mat.values, cmap="RdBu_r", vmin=-lim, vmax=lim,
                        aspect="auto")
    axes[0].set_xticks(range(mat.shape[1]),
                       [f"{c}h" for c in mat.columns])
    axes[0].set_yticks(range(mat.shape[0]), list(mat.index))
    axes[0].set_xlabel("lead time"); axes[0].set_ylabel(
        "observed speed bin (m/s)")
    axes[0].set_title("(a) conditional bias, speed bin × lead")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if v == v:
                axes[0].text(j, i, f"{v:+.1f}", ha="center", va="center",
                             fontsize=6.5,
                             color="white" if abs(v) > 0.6 * lim else "black")
    fig.colorbar(im, ax=axes[0], label="bias (m/s), fcst−obs", shrink=0.9)
    lim2 = np.nanmax(np.abs(ts.values))
    im2 = axes[1].imshow(ts.values, cmap="RdBu_r", vmin=-lim2, vmax=lim2,
                         aspect="auto")
    axes[1].set_xticks(range(ts.shape[1]), list(ts.columns))
    axes[1].set_yticks(range(ts.shape[0]), list(ts.index))
    axes[1].set_title("(b) bias, terrain × season\n(DJF = Dec only)")
    for i in range(ts.shape[0]):
        for j in range(ts.shape[1]):
            v = ts.values[i, j]
            if v == v:
                axes[1].text(j, i, f"{v:+.1f}", ha="center", va="center",
                             fontsize=7,
                             color="white" if abs(v) > 0.6 * lim2 else "black")
    fig.colorbar(im2, ax=axes[1], label="bias (m/s)", shrink=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_bias_structure.png")
    plt.close(fig); made.append("fig2")

    # ---------------- 图3 分lead偏差 + 高度订正 ----------------
    bl = pd.read_csv(E4 / "bias_by_lead.csv").sort_values("lead_h")
    off03 = 1.721  # E4b: z0=0.03 中性对数廓线期望偏移
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.bar(bl.lead_h.astype(str), bl.bias, color="#0173B2",
           label="raw bias (obs ~3 m vs GFS 10 m)")
    ax.plot(bl.lead_h.astype(str), bl.rmse, "^:", ms=4, color="0.35",
            label="RMSE (separate metric)")
    ax.plot(bl.lead_h.astype(str), bl.bias - off03, "o--", color="#D55E00",
            label="height-corrected (z₀=0.03 m, guide)")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("lead time (h)"); ax.set_ylabel("bias (m/s), fcst−obs")
    ax.set_title("Wind-speed bias by lead time")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout(); fig.savefig(OUT / "fig3_bias_lead_height.png",
                                    bbox_inches="tight")
    plt.close(fig); made.append("fig3")

    # ---------------- 图4 (v2: stnMOS/EMOS/阈值曲线; QS3 命名) ----------------
    mae_map = {"GFS_raw": 1.975, "QM": 1.901, "MOS": 1.796,
               "stnMOS": 1.770, "EMOS": 1.802, "LGBM": 1.764}
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.0))
    names = list(mae_map)
    axes[0].bar(names, [mae_map[n] for n in names], color="#0173B2")
    axes[0].set_ylabel("MAE (m/s)")
    axes[0].set_title("(a) point skill (MAE)")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].set_ylim(1.6, 2.1)
    qs = {"EMOS": (1.189, 1.320), "LGBM": (1.111, 1.244)}
    x2 = np.arange(len(qs))
    axes[1].bar(x2 - 0.18, [qs[k][0] for k in qs], width=0.34,
                color="#029E73", label="QS$_3$ (3-quantile)")
    axes[1].bar(x2 + 0.18, [qs[k][1] for k in qs], width=0.34,
                color="#D55E00", label="CRPS (full dist.)")
    axes[1].set_xticks(x2, list(qs))
    axes[1].set_title("(b) distributional scores")
    axes[1].legend(fontsize=6.5)
    tc = pd.read_csv("/home/vhyang/zcode/awf/outputs/review_v2/"
                     "threshold_curves_v2.csv")
    colors = {"GFS": "0.4", "LGBM_q90": "#0173B2", "EMOS_q90": "#D55E00"}
    for name, g in tc.groupby("method"):
        axes[2].plot(g.POFD, g.POD, color=colors.get(name, "k"),
                     label=name.replace("_q90", " q90"))
    axes[2].set_xlabel("POFD (F/(F+C))")
    axes[2].set_ylabel("POD (H/(H+M))")
    axes[2].set_title("(c) strong-wind detection\nthreshold sweep (obs>15 m/s)")
    axes[2].legend(fontsize=6.5, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig4_methods.png",
                                    bbox_inches="tight")
    plt.close(fig); made.append("fig4")

    # ---------------- 图5 LOO 地图 ----------------
    loo2 = loo.merge(reg[["station", "lat", "lon"]], on="station")
    fig, ax = plt.subplots(figsize=(4.9, 4.8))
    ax.set_position([0.03, 0.05, 0.72, 0.9])
    draw_map(ax, segs, "Leave-one-station-out generalization",
             extent=3450e3)
    sc = ax.scatter(*TR.transform(loo2.lon.values, loo2.lat.values),
                    c=loo2.mae_gain_pct, cmap="RdBu_r", vmin=-120, vmax=120,
                    s=26, edgecolor="k", lw=0.3)
    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("LOO MAE gain vs raw GFS (%)")
    ax.set_title("Leave-one-station-out generalization")
    fig.savefig(OUT / "fig5_loo_map.png")
    plt.close(fig); made.append("fig5")

    # ---------------- 图6 消融 ----------------
    ab = pd.read_csv(E5 / "ablation/ablation.csv")
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    order = ["full", "no_terrain", "no_season", "no_lead", "no_geo",
             "gfs_only"]
    ab = ab.set_index("group").loc[order].reset_index()
    ax.bar(ab.group, ab.mae, color="#0173B2")
    for i, (m, c) in enumerate(zip(ab.mae, ab.crps3q)):
        ax.text(i, m + 0.01, f"QS3 {c:.2f}", ha="center", fontsize=6.5)
    ax.set_ylabel("test MAE (m/s)")
    ax.set_title("Feature-group ablation (wind-speed quantiles)")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(OUT / "fig6_ablation.png")
    plt.close(fig); made.append("fig6")

    (OUT / "e6_manifest.json").write_text(json.dumps(dict(
        run_id="AWF_E6_FIGS_R1",
        utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        figures=made, dpi=300,
        sources={"fig2": "E4 bias_by_speedbin/terrain_season",
                 "fig3": "E4 bias_by_lead + E4b R3 offset 1.721 (z0=0.03)",
                 "fig4": "E5 metrics_*.csv + test_predictions.parquet",
                 "fig5": "E5B loo_by_station.csv", "fig6": "E5D ablation.csv"}),
        ensure_ascii=False, indent=2))
    print("figures:", made, "->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
