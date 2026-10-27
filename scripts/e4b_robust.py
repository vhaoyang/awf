#!/usr/bin/env python
"""AWF E4B — E4 判据的稳健性对照（敏感性分析, 独立于预注册 E4 脚本）

运行号: AWF_E4B_ROBUST_R1 (服务器端自编, 非本侧任务号)

背景 (2026-10-23 等待 E2 期间的预检发现, 详见 outputs/smoke 与交付报告):
  R1 机构/高度混杂: AntAWS 无逐站风速计高度(ESSD 正文: 标称~3m, 中山10m,
     1-6m 漂移), 仪器按机构分网络。=> 用"机构内"子样本重做地形对照,
     若地形结构在单一机构内(AMRC, 38 站跨全 5 档)仍成立, 则混杂不致命。
  R2 季节口径: GFS 桶 2021 年 1-2 月/3 月上旬永久无档 => DJF=12月单月,
     MAM=8+61 天。B3 的季节振幅必须按此口径解读, 本脚本输出各季
     天数与样本量的透明分母。
  R3 高度错配量级: obs~3m vs GFS 10m, 对数廓线 (z0=0.01/0.03/0.1 m)
     预期 +15~25% 正偏差 — 对照实测总偏差, 量化其可解释份额。

输入: derived/paired/paired_2021.parquet (E3 产出)
输出: outputs/e4_diag/{robust_institution_terrain.csv, robust_season_denom.csv,
                        e4b_summary.json}
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/mnt/d/data/awfdata")
PAIR = DATA / "derived" / "paired" / "paired_2021.parquet"
REG = DATA / "derived" / "station_qc" / "station_sensor_registry.csv"
OUT = Path("/home/vhyang/zcode/awf/outputs/e4_diag")

ELEV_BINS = [-1, 100, 500, 1000, 2000, 4500]
ELEV_LABELS = ["coast<100", "100-500", "500-1000", "slope1-2k", "plateau>2k"]
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
           6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def log_profile_ratio(z_obs: float, z_fcst: float = 10.0, z0: float = 0.03):
    """中性层结对数廓线风速比 u(z_fcst)/u(z_obs)。"""
    return np.log(z_fcst / z0) / np.log(z_obs / z0)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PAIR)
    reg = pd.read_csv(REG)
    df = df.merge(reg[["station", "institution", "nominal_anem_height_m"]],
                  on="station", how="left")
    df["season"] = df.valid_time.dt.month.map(SEASONS)
    df["fcst_spd"] = np.hypot(df.u10, df.v10)
    df["spd_err"] = df.fcst_spd - df.wspd          # 正=预报偏大 (与 E4 同向)

    # --- R1: 机构 x 地形 条件偏差; 关键=AMRC 机构内地形极差 ---
    # 用站点级均值再聚合, 避免 3-hourly 重复行对站点加权不等
    reg2 = reg.copy()
    reg2["terrain"] = pd.cut(reg2.elevation, ELEV_BINS, labels=ELEV_LABELS)
    site_bias = df.groupby(["station", "institution"], as_index=False).agg(
        bias=("spd_err", "mean"), n=("spd_err", "size"))
    site_bias = site_bias.merge(reg2[["station", "terrain"]], on="station")
    it = site_bias.groupby(["institution", "terrain"], observed=True).agg(
        n_sites=("bias", "size"), n_rows=("n", "sum"),
        bias_mean=("bias", "mean")).round(3).reset_index()
    it.to_csv(OUT / "robust_institution_terrain.csv", index=False)

    amrc = it[it.institution == "AMRC"]
    r1 = dict(
        amrc_terrain_spread_mps=float(amrc.bias_mean.max() - amrc.bias_mean.min())
        if len(amrc) >= 2 else None,
        amrc_bins=len(amrc),
        full_sample_terrain_spread_mps=float(
            site_bias.groupby("terrain", observed=True).bias.mean()
            .pipe(lambda s: s.max() - s.min())),
        note="AMRC 38 站跨 5 地形档; 机构内极差 >= 1.5 m/s 则地形结构"
             "不受机构(仪器/高度网络)混杂解释",
    )

    # --- R2: 季节口径透明分母 ---
    seas = df.groupby("season").agg(
        n=("spd_err", "size"),
        n_days=("valid_time", lambda s: s.dt.date.nunique()),
        bias=("spd_err", "mean")).round(3)
    seas.to_csv(OUT / "robust_season_denom.csv")
    r2 = dict(season_table=seas.reset_index().to_dict("records"),
              djf_caveat="DJF=12月单月(1-2月无档); MAM=3月8天+4/5月",
              season_amp_full=float(seas.bias.max() - seas.bias.min()))

    # --- R3: 高度错配量级 vs 实测总偏差 ---
    obs_mean_spd = float(df.wspd.mean())
    offset_nominal = obs_mean_spd * (log_profile_ratio(3.0, 10.0, 0.03) - 1)
    measured = float(df.spd_err.mean())     # fcst - obs (正=预报偏大)
    r3 = dict(
        obs_mean_spd_mps=obs_mean_spd,
        expected_offset_mps={f"z0={z0}": round(
            obs_mean_spd * (log_profile_ratio(3.0, 10.0, z0) - 1), 3)
            for z0 in (0.01, 0.03, 0.1)},
        measured_bias_mps=measured,
        height_corrected_bias_mps=round(measured - offset_nominal, 3),
        note="spd_err = fcst(10m) - obs(~3m)。若把 obs 沿中性对数廓线订正到 "
             "10m, 偏差 = measured - offset(z0=0.03)。实测为负(GFS 低报)时, "
             "高度订正使低报加深 => 低报信号对高度校正稳健",
    )

    out = dict(
        run_id="AWF_E4B_ROBUST_R1",
        utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        n_rows=len(df), n_stations=int(df.station.nunique()),
        R1_institution_confounding=r1, R2_season_denominator=r2,
        R3_height_mismatch=r3,
    )
    (OUT / "e4b_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str))
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
