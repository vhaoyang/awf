#!/usr/bin/env python
"""Construct GFS-orography predictors and compare early spatial feature sets.

The experiment contrasts coordinate-based and terrain-related inputs using
the configured temporal and held-station evaluations. Differences between
feature sets are predictive comparisons, not proof of a physical mechanism.
Run ID: AWF_E5G_PHYS_R1."""
import os
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xarray as xr

DATA = Path(os.environ.get("AWF_DATA_ROOT", "data"))
PAIR = DATA / "derived" / "paired" / "paired_2021.parquet"
REG = DATA / "derived" / "station_qc" / "station_sensor_registry.csv"
HGT = DATA / "raw" / "gfs" / "gfs_hgt_surface.grib2"
OUT = Path("outputs/e5_baseline/phys")

SEED = 42
THREADS = 24
QUANTILES = [0.1, 0.5, 0.9]
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
         min_data_in_leaf=50, subsample=0.9, subsample_freq=1,
         colsample_bytree=0.9, seed=SEED, num_threads=THREADS,
         deterministic=True, force_col_wise=True, verbosity=-1)
FEATS_BASE = ["u10", "v10", "fcst_spd", "t2m", "prmsl", "rh2m", "lead_h",
              "month_sin", "month_cos", "lat", "lon", "elevation", "terrain"]
PHYS = ["slope", "katab_align"]
CONFIGS = {
    "P1_nocoord_phys": [f for f in FEATS_BASE if f not in ("lat", "lon")] + PHYS,
    "P2_full_phys": FEATS_BASE + PHYS,
}
# 基线 (E5B 已有, 不重跑): pooled MAE 2.110 / CRPS 1.379; 站等权 2.127/1.400
BASELINE_LOO = dict(pooled_mae=2.110, pooled_crps=1.379,
                    stat_eq_mae=2.127, stat_eq_crps=1.400, wins=33)


def circ_deg(d):
    return (d + 180.0) % 360.0 - 180.0


def add_phys(df: pd.DataFrame) -> pd.DataFrame:
    ds = xr.open_dataset(HGT, engine="cfgrib",
                         backend_kwargs={"indexpath": ""})
    name = [v for v in ds.data_vars][0]     # 单消息子集, 唯一变量 (GFS HGT->orog)
    hgt = ds[name]
    if "m**2 s**-2" in str(hgt.attrs.get("units", "")):
        hgt = hgt / 9.80665                  # 位势 -> 米
    lat = hgt.latitude.values
    lon = hgt.longitude.values
    lon = np.where(lon > 180, lon - 360, lon)
    h = hgt.values
    # 网格梯度 (m/m): 两轴坐标都换成米 (弧度 x 地球半径), 经向再除 cos(lat)
    R = 6.371e6
    y_m = np.deg2rad(lat) * R
    x_m = np.deg2rad(lon) * R
    gy, gx = np.gradient(h, y_m, x_m)
    gx = gx / np.maximum(np.cos(np.deg2rad(lat))[:, None], 0.1)
    gxi = xr.DataArray(gx, coords={"latitude": lat, "longitude": lon})
    gyi = xr.DataArray(gy, coords={"latitude": lat, "longitude": lon})
    st_lat = xr.DataArray(df.lat.values, dims="station")
    st_lon = xr.DataArray(df.lon.values, dims="station")
    gx_s = gxi.interp(latitude=st_lat, longitude=st_lon).values
    gy_s = gyi.interp(latitude=st_lat, longitude=st_lon).values
    slope = np.hypot(gx_s, gy_s) * 1000.0          # m per km
    # katabatic 对齐: 预报风矢量 与 下坡向量 (−gx,−gy) 的余弦 (逐行)
    vdot = (-gx_s) * df.u10.values + (-gy_s) * df.v10.values
    norm = np.hypot(gx_s, gy_s) * np.maximum(np.hypot(df.u10, df.v10).values,
                                             1e-6)
    df["slope"] = np.broadcast_to(slope, df.u10.shape)
    df["katab_align"] = vdot / norm
    ds.close()
    return df


def loo(df, feats):
    rows = []
    t0 = time.time()
    for k, s in enumerate(sorted(df.station.unique()), 1):
        tr = df[df.station != s]
        te = df[df.station == s]
        y = te.wspd.values
        preds = {}
        for tau in QUANTILES:
            m = lgb.LGBMRegressor(objective="quantile", alpha=tau, **P)
            m.fit(tr[feats], tr.wspd, categorical_feature=[
                feats.index("terrain")] if "terrain" in feats else None)
            preds[tau] = m.predict(te[feats])
        pin = []
        for tau in QUANTILES:
            z = y - preds[tau]
            pin.append(np.mean(np.maximum(tau * z, (tau - 1) * z)))
        rows.append(dict(station=s, n=len(te),
                         lgbm_mae=float(np.abs(preds[0.5] - y).mean()),
                         lgbm_crps3q=float(2 * np.mean(pin))))
        if k % 10 == 0:
            print(f"  fold {k}/50 ({time.time()-t0:.0f}s)", flush=True)
    out = pd.DataFrame(rows)
    w = out.n / out.n.sum()
    return dict(pooled_mae=float((out.lgbm_mae * w).sum()),
                pooled_crps=float((out.lgbm_crps3q * w).sum()),
                stat_eq_mae=float(out.lgbm_mae.mean()),
                stat_eq_crps=float(out.lgbm_crps3q.mean())), out


def temporal(df, feats):
    tr = df[df.month <= 8]
    va = df[df.month == 9]
    te = df[df.month >= 10]
    y = te.wspd.values
    preds = {}
    for tau in QUANTILES:
        m = lgb.LGBMRegressor(objective="quantile", alpha=tau,
                              **{**P, "n_estimators": 400})
        m.fit(tr[feats], tr.wspd, categorical_feature=[
            feats.index("terrain")] if "terrain" in feats else None,
            eval_set=[(va[feats], va.wspd)], eval_metric="quantile",
            callbacks=[lgb.early_stopping(50, verbose=False)])
        preds[tau] = m.predict(te[feats], num_iteration=m.best_iteration_)
    pin = [np.mean(np.maximum(tau * (y - preds[tau]),
                              (tau - 1) * (y - preds[tau])))
           for tau in QUANTILES]
    return dict(mae=float(np.abs(preds[0.5] - y).mean()),
                crps3q=float(2 * np.mean(pin)))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = "AWF_E5G_PHYS_R1"
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
    df = add_phys(df)
    print(f"[{run_id}] slope m/km: p50={np.median(df.slope):.2f} "
          f"p95={np.percentile(df.slope, 95):.2f}; "
          f"katab_align mean={df.katab_align.mean():.3f}")

    summary = dict(
        run_id=run_id, utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        n_rows=len(df),
        baseline_loo_A_full=BASELINE_LOO,
        slope_note="GFS HGT 0.25° 网格梯度双线性插值到站; katab_align=风矢量·下坡向量/(|w||∇H|)",
        configs={},
    )
    all_rows = []
    for name, feats in CONFIGS.items():
        print(f"[{run_id}] LOO {name} ...", flush=True)
        agg, out = loo(df, feats)
        out["config"] = name
        all_rows.append(out)
        tc = temporal(df, feats)
        agg["temporal_mae"] = tc["mae"]
        agg["temporal_crps3q"] = tc["crps3q"]
        summary["configs"][name] = agg
        print(f"  {name}: pooled MAE {agg['pooled_mae']:.3f} "
              f"CRPS {agg['pooled_crps']:.3f} | temporal "
              f"MAE {tc['mae']:.3f} CRPS {tc['crps3q']:.3f}", flush=True)

    pd.concat(all_rows, ignore_index=True).round(3).to_csv(
        OUT / "loo_phys_by_station.csv", index=False)
    (OUT / "loo_phys_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary["configs"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
