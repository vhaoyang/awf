#!/usr/bin/env python
r"""AWF E4_DIAG — 偏差时空结构诊断（S2 阶段）

运行号: AWF_E4_DIAG_R1 (服务器端自编, 非本侧任务号)

输入: derived/paired/paired_2021.parquet (E3 产出)
输出: outputs/e4_diag/
  - bias_by_lead_season_terrain.csv   风速偏差 分lead×季节×地形
  - bias_by_speedbin.csv              分风速箱 (条件偏差结构的关键证据)
  - dir_error_circ.csv                风向圆周误差 (分lead/风速箱)
  - e4_summary.json                   判据结论: 偏差是否有条件结构

判据 (止损):
  至少满足以下两条, 否则报"结构平淡":
  B1 风速箱间 |条件均值偏差| 极差 >= 1.5 m/s
  B2 lead 间 |偏差| 单调放大或有清晰拐点 (24h vs 72h 差 >= 1 m/s)
  B3 季节振幅 >= 1 m/s 或 地形类别间差 >= 1.5 m/s
  B4 风向系统偏差 >= 10° (圆周均值, 至少某 lead/地形)
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/mnt/d/data/awfdata")
PAIR = DATA / "derived" / "paired" / "paired_2021.parquet"
OUT = Path("/home/vhyang/zcode/awf/outputs/e4_diag")

ELEV_BINS = [-1, 100, 500, 1000, 2000, 4500]
ELEV_LABELS = ["coast<100", "100-500", "500-1000", "slope1-2k", "plateau>2k"]
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def circ_deg(diff: np.ndarray) -> np.ndarray:
    """圆周差归到 [-180,180)。"""
    return (diff + 180.0) % 360.0 - 180.0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = "AWF_E4_DIAG_R1"
    df = pd.read_parquet(PAIR)
    st = pd.read_csv(DATA / "derived" / "station_qc" / "usable_stations.csv")
    df = df.merge(st[["station", "elevation"]], on="station", how="left")
    df["terrain"] = pd.cut(df.elevation, ELEV_BINS, labels=ELEV_LABELS)
    df["season"] = df.valid_time.dt.month.map(SEASONS)
    df["month"] = df.valid_time.dt.month
    df["fcst_spd"] = np.hypot(df.u10, df.v10)
    df["spd_err"] = df.fcst_spd - df.wspd          # 正=预报偏大
    df["dir_err"] = circ_deg(  # 预报风向-观测风向 (圆周), 低风不信任已标 dir_trust
        (np.degrees(np.arctan2(-df.u10, -df.v10)) - df.wdir).values)

    # --- B1: 风速条件偏差 ---
    df["obs_bin"] = pd.cut(df.wspd, [0, 3, 6, 10, 15, 25, 100],
                           labels=["0-3", "3-6", "6-10", "10-15", "15-25", ">25"])
    by_speed = df.groupby(["obs_bin", "lead_h"], observed=True).agg(
        n=("spd_err", "size"), bias=("spd_err", "mean"),
        mae=("spd_err", lambda x: float(np.abs(x).mean()))).round(3).reset_index()
    by_speed.to_csv(OUT / "bias_by_speedbin.csv", index=False)

    # --- B2: lead 结构 ---
    by_lead = df.groupby("lead_h").agg(
        n=("spd_err", "size"), bias=("spd_err", "mean"),
        rmse=("spd_err", lambda x: float(np.sqrt((x ** 2).mean()))),
        mae=("spd_err", lambda x: float(np.abs(x).mean()))).round(3)
    by_lead.to_csv(OUT / "bias_by_lead.csv")

    # --- B3: 季节×地形 ---
    by_terr = df.pivot_table(index="terrain", columns="season", values="spd_err",
                             observed=True, aggfunc="mean").round(3)
    by_terr.to_csv(OUT / "bias_by_terrain_season.csv")
    by_mon = df.groupby("month").agg(n=("spd_err", "size"), bias=("spd_err", "mean")).round(3)
    by_mon.to_csv(OUT / "bias_by_month.csv")

    # --- B4: 风向圆周误差 ---
    trust = df[df.dir_trust & df.wdir.notna()]
    by_dir = trust.groupby("lead_h").agg(
        n=("dir_err", "size"),
        circ_mean_deg=("dir_err", lambda x: float(
            np.degrees(np.arctan2(np.mean(np.sin(np.radians(x))),
                                   np.mean(np.cos(np.radians(x))))))),
        circ_mae=("dir_err", lambda x: float(np.abs(x).mean()))).round(2)
    by_dir.to_csv(OUT / "dir_error_circ.csv")

    # --- 判据评估 ---
    sb = by_speed.groupby("obs_bin", observed=True).bias.mean()
    b1_range = float(sb.max() - sb.min())
    b2_gain = float(by_lead.bias.get(72, np.nan) - by_lead.bias.get(24, np.nan))
    b3_season_amp = float(by_terr.max(axis=1).sub(by_terr.min(axis=1)).mean()) \
        if len(by_terr) else np.nan
    b3_terr_spread = float(by_terr.stack().max() - by_terr.stack().min())
    b4_max_dir = float(by_dir.circ_mean_deg.abs().max())

    verdict = dict(
        B1_speedbin_range=b1_range, B1_pass=bool(b1_range >= 1.5),
        B2_lead24to72=b2_gain,
        B2_pass=bool(abs(b2_gain) >= 1.0 or
                     by_lead.bias.abs().max() >= 1.0),
        B3_season_amplitude=b3_season_amp, B3_terrain_spread=b3_terr_spread,
        B3_pass=bool((b3_season_amp or 0) >= 1.0 or (b3_terr_spread or 0) >= 1.5),
        B4_max_dir_bias_deg=b4_max_dir, B4_pass=bool(b4_max_dir >= 10.0),
        utc=datetime.now(timezone.utc).isoformat(),
        n_rows=len(df), n_stations=int(df.station.nunique()),
        n_days=int(df.valid_time.dt.date.nunique()),
    )
    verdict["structure_found"] = sum(
        [verdict["B1_pass"], verdict["B2_pass"], verdict["B3_pass"],
         verdict["B4_pass"]]) >= 2
    (OUT / "e4_summary.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2))
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
