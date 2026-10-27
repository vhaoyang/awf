#!/usr/bin/env python
"""AWF E5H — REMA 高分辨率地形物理特征与 LOO 泛化检验（E5G 完整版）

运行号: AWF_E5H_REMA_R1 (服务器端自编, 非本侧任务号)

动机: E5G 两个粗特征(GFS 0.25° 坡度+katabatic 对齐)已把 LOO 未见站点增益
从 −1.0% 提到 −4.1%。本实验用 REMA 1km 镶嵌构建高分辨率地形特征组,
检验能否继续推进泛化。

特征 (站级静态 + 逐行动态):
  slope1k / slope5k / slope25k   REMA 梯度模长, 1km 原生 + 5/25km 高斯平滑窗 (m/km)
  prominence                     站点高程 − 25km 半径均值高程 (m)
  fetch_drop                     沿陡降方向(D8)走 50km 的累计落差 (m, katabatic 代理)
  katab_align_rema               逐行: 预报风矢量 与 REMA 5km 尺度下坡向量 夹角余弦

配置 (LOO, 与 E5B/E5G 同协议):
  P1r no-coords + REMA 物理组   (与 E5G-P1 同构, 特换特征组)
  P2r full13 + REMA 物理组
对照 (不重跑): A_full LOO pooled MAE 2.110 / CRPS 1.379 / wins 33;
  E5G-P1 pooled 2.044 / 1.340 / wins 38。另跑时间块检查站网内不塌。

输出: outputs/e5_baseline/rema/{rema_features_by_station.csv,
       loo_rema_summary.json, loo_rema_by_station.csv}
注意: LightGBM 串行 (24 线程单进程)。
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
import rasterio
from pyproj import CRS, Transformer
from scipy.ndimage import gaussian_filter, map_coordinates

DATA = Path("/mnt/d/data/awfdata")
PAIR = DATA / "derived" / "paired" / "paired_2021.parquet"
REG = DATA / "derived" / "station_qc" / "station_sensor_registry.csv"
REMA = DATA / "raw" / "rema" / "rema_mosaic_1km_v2.0_dem.tif"
OUT = Path("/home/vhyang/zcode/awf/outputs/e5_baseline/rema")

SEED = 42
THREADS = 24
QUANTILES = [0.1, 0.5, 0.9]
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
         min_data_in_leaf=50, subsample=0.9, subsample_freq=1,
         colsample_bytree=0.9, seed=SEED, num_threads=THREADS,
         deterministic=True, force_col_wise=True, verbosity=-1)
FEATS_BASE = ["u10", "v10", "fcst_spd", "t2m", "prmsl", "rh2m", "lead_h",
              "month_sin", "month_cos", "lat", "lon", "elevation", "terrain"]
REMA_STATIC = ["slope1k", "slope5k", "slope25k", "prominence", "fetch_drop"]
REMA_DYN = ["katab_align_rema"]
CONFIGS = {
    "P1r_nocoord_rema": [f for f in FEATS_BASE
                         if f not in ("lat", "lon")] + REMA_STATIC + REMA_DYN,
    "P2r_full_rema": FEATS_BASE + REMA_STATIC + REMA_DYN,
}
REF = dict(A_full=dict(pooled_mae=2.110, pooled_crps=1.379, wins=33),
           E5G_P1=dict(pooled_mae=2.044, pooled_crps=1.340, wins=38))


def load_rema():
    ds = rasterio.open(REMA)
    elev = ds.read(1).astype(np.float32)
    nodata = ds.nodata
    if nodata is not None:
        elev[elev == nodata] = np.nan
    elev[elev < -100] = np.nan          # 海洋填充
    tr, crs = ds.transform, ds.crs
    res = float(np.abs(tr.a))           # 米/像素 (~1000)
    to_map = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    return elev, tr, res, to_map


def station_feats(elev, tr, res, to_map, st: pd.DataFrame) -> pd.DataFrame:
    ny, nx = elev.shape
    xs, ys = to_map.transform(st.lon.values, st.lat.values)
    cols = (xs - tr.c) / tr.a
    rows = (ys - tr.f) / tr.e
    cx = np.clip(np.round(cols).astype(int), 0, nx - 1)
    cy = np.clip(np.round(rows).astype(int), 0, ny - 1)

    # 高斯平滑 sigma 以像素计 (res 单位米)
    sm5 = gaussian_filter(elev, 5000.0 / res)
    sm25 = gaussian_filter(elev, 25000.0 / res)

    def grad(g):
        gy, gx = np.gradient(g, res, res)   # m/m
        return gx, gy

    out = {}
    for name, g in [("1k", elev), ("5k", sm5), ("25k", sm25)]:
        gx, gy = grad(g)
        out[f"slope{name}"] = np.hypot(gx[cy, cx], gy[cy, cx]) * 1000  # m/km
    gx5, gy5 = grad(sm5)
    # prominence: 站高程 − 25km 窗均值 (高斯均值近似)
    out["prominence"] = elev[cy, cx] - sm25[cy, cx]
    # fetch_drop: D8 陡降 50km 累计落差
    drops = []
    for i in range(len(st)):
        y, x = cy[i], cy[i]
        z = elev[cy[i], cx[i]]
        drop = 0.0
        for _ in range(int(50000 // res)):
            y0, y1 = max(y - 1, 0), min(y + 2, ny)
            x0, x1 = max(x - 1, 0), min(x + 2, nx)
            win = elev[y0:y1, x0:x1]
            if not np.isfinite(win).any():
                break
            j = np.nanargmax(win)
            dy, dx = np.unravel_index(j, win.shape)
            zn = win[dy, dx]
            if not np.isfinite(zn) or zn >= z:
                break
            drop += z - zn
            z = zn
            y, x = y0 + dy, x0 + dx
        drops.append(drop)
    out["fetch_drop"] = np.array(drops)
    # 下坡方向 (5km 尺度) 供逐行对齐用
    downslope_u = -gx5[cy, cx]
    downslope_v = -gy5[cy, cx]
    feats = pd.DataFrame(out, index=st.index)
    feats["station"] = st.station.values
    return feats, downslope_u, downslope_v


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = "AWF_E5H_REMA_R1"
    elev, tr, res, to_map = load_rema()
    print(f"[{run_id}] REMA {elev.shape} res={res:.0f}m "
          f"valid={np.isfinite(elev).mean():.2f}")

    df = pd.read_parquet(PAIR).rename(columns={"t2m_x": "obs_t2m",
                                               "t2m_y": "t2m"})
    reg = pd.read_csv(REG)
    df = df.merge(reg[["station", "lat", "lon", "elevation", "terrain"]],
                  on="station", how="left")
    df["terrain"] = df.terrain.astype("category")
    df["fcst_spd"] = np.hypot(df.u10, df.v10)
    df["month"] = df.valid_time.dt.month
    df["month_sin"] = np.sin(2 * np.pi * df.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.month / 12)
    df = df.dropna(subset=["wspd"]).reset_index(drop=True)

    st = reg.drop_duplicates("station")[["station", "lat", "lon"]]
    feats, du, dv = station_feats(elev, tr, res, to_map, st)
    feats.round(3).to_csv(OUT / "rema_features_by_station.csv", index=False)
    print(feats.drop(columns="station").quantile([0.5, 0.95]).round(2).to_string())

    fmap = feats.set_index("station")
    df = df.merge(fmap, on="station", how="left")
    # 逐行 katabatic 对齐 (REMA 5km 下坡方向)
    du_r = df.station.map(dict(zip(st.station, du))).values
    dv_r = df.station.map(dict(zip(st.station, dv))).values
    vdot = du_r * df.u10.values + dv_r * df.v10.values
    norm = np.hypot(du_r, dv_r) * np.maximum(
        np.hypot(df.u10, df.v10).values, 1e-6)
    df["katab_align_rema"] = vdot / np.maximum(norm, 1e-6)

    def loo(fr, feats):
        rows = []
        t0 = time.time()
        for k, s in enumerate(sorted(fr.station.unique()), 1):
            trn = fr[fr.station != s]
            te = fr[fr.station == s]
            y = te.wspd.values
            preds = {}
            for tau in QUANTILES:
                m = lgb.LGBMRegressor(objective="quantile", alpha=tau, **P)
                m.fit(trn[feats], trn.wspd, categorical_feature=[
                    feats.index("terrain")] if "terrain" in feats else None)
                preds[tau] = m.predict(te[feats])
            pin = [np.mean(np.maximum(tau * (y - preds[tau]),
                                      (tau - 1) * (y - preds[tau])))
                   for tau in QUANTILES]
            rows.append(dict(station=s, n=len(te),
                             lgbm_mae=float(np.abs(preds[0.5] - y).mean()),
                             lgbm_crps3q=float(2 * np.mean(pin))))
            if k % 10 == 0:
                print(f"  fold {k}/50 ({time.time()-t0:.0f}s)", flush=True)
        o = pd.DataFrame(rows)
        w = o.n / o.n.sum()
        return dict(pooled_mae=float((o.lgbm_mae * w).sum()),
                    pooled_crps=float((o.lgbm_crps3q * w).sum()),
                    stat_eq_mae=float(o.lgbm_mae.mean())), o

    def temporal(fr, feats):
        trn = fr[fr.month <= 8]
        va = fr[fr.month == 9]
        te = fr[fr.month >= 10]
        y = te.wspd.values
        preds = {}
        for tau in QUANTILES:
            m = lgb.LGBMRegressor(
                objective="quantile", alpha=tau, **{**P, "n_estimators": 400})
            m.fit(trn[feats], trn.wspd, categorical_feature=[
                feats.index("terrain")] if "terrain" in feats else None,
                eval_set=[(va[feats], va.wspd)], eval_metric="quantile",
                callbacks=[lgb.early_stopping(50, verbose=False)])
            preds[tau] = m.predict(te[feats],
                                   num_iteration=m.best_iteration_)
        pin = [np.mean(np.maximum(tau * (y - preds[tau]),
                                  (tau - 1) * (y - preds[tau])))
               for tau in QUANTILES]
        return dict(mae=float(np.abs(preds[0.5] - y).mean()),
                    crps3q=float(2 * np.mean(pin)))

    summary = dict(run_id=run_id,
                   utc=datetime.now(timezone.utc).isoformat(),
                   script_sha256=hashlib.sha256(
                       Path(__file__).read_bytes()).hexdigest(),
                   n_rows=len(df), references=REF, configs={})
    gfs_loo = pd.read_csv(Path("/home/vhyang/zcode/awf/outputs/"
                               "e5_baseline/loo/loo_by_station.csv"))
    all_rows = []
    for name, feats in CONFIGS.items():
        print(f"[{run_id}] LOO {name} ...", flush=True)
        agg, o = loo(df, feats)
        o["config"] = name
        m = o.merge(gfs_loo[["station", "gfs_mae"]], on="station")
        agg["wins"] = int((m.lgbm_mae < m.gfs_mae).sum())
        tc = temporal(df, feats)
        agg.update({f"temporal_{k}": v for k, v in tc.items()})
        summary["configs"][name] = agg
        all_rows.append(o)
        print(f"  {name}: pooled {agg['pooled_mae']:.3f} "
              f"CRPS {agg['pooled_crps']:.3f} wins {agg['wins']}/50 "
              f"temporal {tc['mae']:.3f}/{tc['crps3q']:.3f}", flush=True)

    pd.concat(all_rows, ignore_index=True).round(3).to_csv(
        OUT / "loo_rema_by_station.csv", index=False)
    (OUT / "loo_rema_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary["configs"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
