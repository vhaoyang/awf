#!/usr/bin/env python
"""AWF E7 — LOO 同源重跑（修复后地形对齐度 + 唯一键预测交付 + 版本差异）

运行号: AWF_E7_LOO_R1 (服务器端自编, 非本侧任务号)

对应 V2-05(后半)/V2-06:
  - katab_align_EN: 用 e7_terrain 的当地东北基底下坡向量(解析验证过)与
    GFS u10/v10(同为东北分量)直接点积——坐标系一致。
  - 配置: A_full(与历史可比) 与 P1r_fix(去经纬+REMA标量特征+修复对齐度+flat标志)
  - 预测带 station/init/valid_time/lead_h/obs/gfs/q10/q50/q90 —— 指标同源
  - 版本差异表: E5H(2.004, 旧对齐) vs E6C(2.023, 旧对齐) vs 本运行的差异

输出: outputs/review_v2/{loo_e7_pred_A_full.parquet, loo_e7_pred_P1r_fix.parquet,
       loo_e7_summary.json}
"""
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from e7_core import monotonize_rows

DATA = Path("/mnt/d/data/awfdata")
PAIR = DATA / "derived" / "paired" / "paired_2021.parquet"
REMAF = Path("/home/vhyang/zcode/awf/outputs/e5_baseline/rema/"
             "rema_features_by_station.csv")
TFV = Path("/home/vhyang/zcode/awf/outputs/review_v2/terrain_features_v2.csv")
TFJ = Path("/home/vhyang/zcode/awf/outputs/review_v2/terrain_validation.json")
OUT = Path("/home/vhyang/zcode/awf/outputs/review_v2")
QS = [0.1, 0.5, 0.9]
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
         min_data_in_leaf=50, subsample=0.9, subsample_freq=1,
         colsample_bytree=0.9, seed=42, num_threads=24,
         deterministic=True, force_col_wise=True, verbosity=-1)
FEATS_A = ["u10", "v10", "fcst_spd", "t2m", "prmsl", "rh2m", "lead_h",
           "month_sin", "month_cos", "lat", "lon", "elevation", "terrain"]
REMA_STATIC = ["slope1k", "slope5k", "slope25k", "prominence", "fetch_drop"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = "AWF_E7_LOO_R1"
    df = pd.read_parquet(PAIR).rename(columns={"t2m_x": "obs_t2m",
                                               "t2m_y": "t2m"})
    reg = pd.read_csv(DATA / "derived/station_qc/station_sensor_registry.csv")
    df = df.merge(reg[["station", "lat", "lon", "elevation", "terrain"]],
                  on="station", how="left")
    df["terrain"] = df.terrain.astype("category")
    df["fcst_spd"] = np.hypot(df.u10, df.v10)
    df["month"] = df.valid_time.dt.month
    df["month_sin"] = np.sin(2 * np.pi * df.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.month / 12)
    df = df.dropna(subset=["wspd"]).reset_index(drop=True)
    df = df.merge(pd.read_csv(REMAF), on="station", how="left")

    # 修复后的对齐度: terrain_validation.json 有逐站验证; 下坡向量需重算
    import rasterio
    from pyproj import CRS, Transformer
    from scipy.ndimage import gaussian_filter
    ds = rasterio.open(DATA / "raw" / "rema" / "rema_mosaic_1km_v2.0_dem.tif")
    elev = ds.read(1).astype(np.float32)
    if ds.nodata is not None:
        elev[elev == ds.nodata] = np.nan
    elev[elev < -100] = np.nan
    tr, crs = ds.transform, ds.crs
    y_ax = tr.f + np.arange(elev.shape[0]) * tr.e
    x_ax = tr.c + np.arange(elev.shape[1]) * tr.a
    sm5 = gaussian_filter(np.nan_to_num(elev, nan=0.0), 5.0)
    gy, gx = np.gradient(sm5, y_ax, x_ax)       # 有符号坐标
    to_map = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    stq = reg.drop_duplicates("station")[["station", "lat", "lon"]]
    downslope_EN = {}
    flat = {}
    for _, s in stq.iterrows():
        mx, my = to_map.transform(s.lon, s.lat)
        ci = int(round((mx - tr.c) / tr.a))
        ri = int(round((my - tr.f) / tr.e))
        if not (0 <= ri < elev.shape[0] and 0 <= ci < elev.shape[1]) \
                or not np.isfinite(elev[ri, ci]):
            downslope_EN[s.station] = (np.nan, np.nan)
            flat[s.station] = True
            continue
        # 当地东/北单位向量(pyproj 经验)
        x0, y0 = mx, my
        ex, ey = to_map.transform(s.lon + 0.05 / np.cos(np.deg2rad(s.lat)),
                                  s.lat)
        ev = np.array([ex - x0, ey - y0]); ev /= np.hypot(*ev)
        nx_, ny_ = to_map.transform(s.lon, s.lat + 0.05)
        nv = np.array([nx_ - x0, ny_ - y0]); nv /= np.hypot(*nv)
        gx_s, gy_s = gx[ri, ci], gy[ri, ci]
        gE = gx_s * ev[0] + gy_s * ev[1]
        gN = gx_s * nv[0] + gy_s * nv[1]
        nrm = np.hypot(gE, gN)
        if nrm < 1e-9:
            downslope_EN[s.station] = (0.0, 0.0)
            flat[s.station] = True
        else:
            downslope_EN[s.station] = (-gE / nrm, -gN / nrm)
            flat[s.station] = False
    du = df.station.map(lambda s: downslope_EN[s][0]).values
    dv = df.station.map(lambda s: downslope_EN[s][1]).values
    nrm_w = np.maximum(np.hypot(df.u10, df.v10).values, 1e-6)
    df["katab_align_EN"] = (du * df.u10.values + dv * df.v10.values) / nrm_w
    df["is_flat"] = df.station.map(flat).astype(int)
    # flat 站对齐度置0(不伪造方向)
    df.loc[df.is_flat == 1, "katab_align_EN"] = 0.0

    FEATS_P1r = [f for f in FEATS_A if f not in ("lat", "lon")] + \
        REMA_STATIC + ["katab_align_EN", "is_flat"]
    summary = dict(run_id=run_id,
                   utc=datetime.now(timezone.utc).isoformat(),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()
                                                ).hexdigest(),
                   configs={}, version_diff={
                       "E5H_P1r_old": "pooled 2.004 (旧对齐度: 投影梯度直点积)",
                       "E6C_P1r": "pooled 2.023 (旧对齐度, 存预测复算)",
                       "E7_P1r_fix": "本运行: 当地东北基底修复对齐度",
                       "差异原因": ("E5H 与 E6C 均含坐标基底错误; 两者间差异"
                                    "来自特征构造细节(见各自脚本哈希); "
                                    "本表以 E7 修复版为准")})
    for cname, feats in (("A_full", FEATS_A), ("P1r_fix", FEATS_P1r)):
        t0 = time.time()
        parts = []
        for s in sorted(df.station.unique()):
            trn = df[df.station != s]
            te = df[df.station == s]
            Q = np.empty((len(te), len(QS)))
            for j, tau in enumerate(QS):
                m = lgb.LGBMRegressor(objective="quantile", alpha=tau, **P)
                m.fit(trn[feats], trn.wspd,
                      categorical_feature=[feats.index("terrain")])
                Q[:, j] = m.predict(te[feats])
            Q = monotonize_rows(Q)
            p = te[["station", "init", "valid_time", "lead_h", "wspd",
                    "fcst_spd"]].copy().rename(
                columns={"wspd": "obs", "fcst_spd": "gfs"})
            p["q10"], p["q50"], p["q90"] = Q[:, 0], Q[:, 1], Q[:, 2]
            parts.append(p)
        pred = pd.concat(parts, ignore_index=True)
        pred.to_parquet(OUT / f"loo_e7_pred_{cname}.parquet", index=False)
        y = pred.obs.values
        pin = [np.mean(np.maximum(t * (y - pred[f"q{int(t*100)}"]),
                                  (t - 1) * (y - pred[f"q{int(t*100)}"])))
               for t in QS]
        summary["configs"][cname] = dict(
            n_rows=len(pred),
            pooled_mae=round(float(np.abs(pred.q50 - y).mean()), 4),
            pooled_gfs_mae=round(float(np.abs(pred.gfs - y).mean()), 4),
            qs3=round(float(2 * np.mean(pin)), 4),
            cov80=round(float(((y >= pred.q10) & (y <= pred.q90)).mean()), 4),
            cov_q90=round(float((y < pred.q90).mean()), 4),
            elapsed_s=round(time.time() - t0, 1))
        print(f"[{run_id}] {cname}: "
              f"{summary['configs'][cname]['pooled_mae']} "
              f"(GFS {summary['configs'][cname]['pooled_gfs_mae']}) "
              f"cov80 {summary['configs'][cname]['cov80']} "
              f"({time.time()-t0:.0f}s)", flush=True)
    (OUT / "loo_e7_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary["configs"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
