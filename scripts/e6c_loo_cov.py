#!/usr/bin/env python
"""AWF E6C — 评审 P0：LOO 分位覆盖与区间诊断（R1.2）

运行号: AWF_E6C_LOOCOV_R1 (服务器端自编, 非本侧任务号)

依据 R1.2: "LOO CRPS3 下降 35% 不等于校准可迁移"。重跑 LOO 保存逐行
q10/50/90 预测(A_full 与 P1r 两配置), 输出:
  逐站/逐 lead 的名义 0.1/0.5/0.9 覆盖率; 80% 中央区间覆盖与平均宽度;
  风速分层覆盖; 与时间块口径(图4b)分开呈现。
输出: outputs/review_p0/{loo_pred_A.parquet, loo_pred_P1r.parquet,
       loo_coverage_by_station.csv, loo_coverage_by_lead.csv,
       loo_cov_summary.json}
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
from scipy.ndimage import gaussian_filter

DATA = Path("/mnt/d/data/awfdata")
PAIR = DATA / "derived" / "paired" / "paired_2021.parquet"
REG = DATA / "derived" / "station_qc" / "station_sensor_registry.csv"
REMA = DATA / "raw" / "rema" / "rema_mosaic_1km_v2.0_dem.tif"
OUT = Path("/home/vhyang/zcode/awf/outputs/review_p0")

SEED = 42
QUANTILES = [0.1, 0.5, 0.9]
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
         min_data_in_leaf=50, subsample=0.9, subsample_freq=1,
         colsample_bytree=0.9, seed=SEED, num_threads=24,
         deterministic=True, force_col_wise=True, verbosity=-1)
FEATS_BASE = ["u10", "v10", "fcst_spd", "t2m", "prmsl", "rh2m", "lead_h",
              "month_sin", "month_cos", "lat", "lon", "elevation", "terrain"]
REMA_STATIC = ["slope1k", "slope5k", "slope25k", "prominence", "fetch_drop"]
CONFIGS = {
    "A_full": FEATS_BASE,
    "P1r": [f for f in FEATS_BASE if f not in ("lat", "lon")]
    + REMA_STATIC + ["katab_align_rema"],
}


def build_df():
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
    feats = pd.read_csv("/home/vhyang/zcode/awf/outputs/e5_baseline/rema/"
                        "rema_features_by_station.csv")
    df = df.merge(feats, on="station", how="left")
    # katab_align_rema (与 E5H 同法)
    ds = rasterio.open(REMA)
    elev = ds.read(1).astype(np.float32)
    if ds.nodata is not None:
        elev[elev == ds.nodata] = np.nan
    elev[elev < -100] = np.nan
    tr, crs, res = ds.transform, ds.crs, float(np.abs(ds.transform.a))
    sm5 = gaussian_filter(elev, 5000.0 / res)
    gy, gx = np.gradient(sm5, res, res)
    st = reg.drop_duplicates("station")[["station", "lat", "lon"]]
    to_map = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    xs, ys = to_map.transform(st.lon.values, st.lat.values)
    cx = np.clip(np.round((xs - tr.c) / tr.a).astype(int), 0,
                 elev.shape[1] - 1)
    cy = np.clip(np.round((ys - tr.f) / tr.e).astype(int), 0,
                 elev.shape[0] - 1)
    du, dv = -gx[cy, cx], -gy[cy, cx]
    m = dict(zip(st.station, zip(du, dv)))
    du_r = df.station.map(lambda s: m[s][0]).values
    dv_r = df.station.map(lambda s: m[s][1]).values
    vdot = du_r * df.u10.values + dv_r * df.v10.values
    norm = np.hypot(du_r, dv_r) * np.maximum(
        np.hypot(df.u10, df.v10).values, 1e-6)
    df["katab_align_rema"] = vdot / np.maximum(norm, 1e-6)
    return df


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = "AWF_E6C_LOOCOV_R1"
    df = build_df()
    summary = dict(run_id=run_id,
                   utc=datetime.now(timezone.utc).isoformat(),
                   script_sha256=hashlib.sha256(
                       Path(__file__).read_bytes()).hexdigest(),
                   configs={})
    by_station_rows, by_lead_rows = [], []
    for cname, feats in CONFIGS.items():
        print(f"[{run_id}] LOO {cname} (saving predictions) ...",
              flush=True)
        t0 = time.time()
        parts = []
        for s in sorted(df.station.unique()):
            trn = df[df.station != s]
            te = df[df.station == s]
            q = {}
            for tau in QUANTILES:
                mmod = lgb.LGBMRegressor(objective="quantile", alpha=tau,
                                         **P)
                mmod.fit(trn[feats], trn.wspd,
                         categorical_feature=[feats.index("terrain")])
                q[tau] = mmod.predict(te[feats])
            parts.append(pd.DataFrame(dict(
                station=s, lead_h=te.lead_h.values, wspd=te.wspd.values,
                q10=q[0.1], q50=q[0.5], q90=q[0.9])))
        pred = pd.concat(parts, ignore_index=True)
        pred.to_parquet(OUT / f"loo_pred_{cname.replace('_full','_A')}"
                        ".parquet" if cname == "A_full" else
                        OUT / "loo_pred_P1r.parquet", index=False)
        y = pred.wspd.values
        cov10 = float((y < pred.q10).mean())
        cov50 = float((y < pred.q50).mean())
        cov90 = float((y < pred.q90).mean())
        in80 = float(((y >= pred.q10) & (y <= pred.q90)).mean())
        width80 = float((pred.q90 - pred.q10).mean())
        summary["configs"][cname] = dict(
            n=len(pred), coverage_q10=cov10, coverage_q50=cov50,
            coverage_q90=cov90, interval80_coverage=in80,
            interval80_mean_width=width80,
            elapsed_s=round(time.time() - t0, 1))
        for s, g in pred.groupby("station"):
            by_station_rows.append(dict(
                config=cname, station=s, n=len(g),
                cov_q10=round(float((g.wspd < g.q10).mean()), 3),
                cov_q50=round(float((g.wspd < g.q50).mean()), 3),
                cov_q90=round(float((g.wspd < g.q90).mean()), 3),
                cov80=round(float(((g.wspd >= g.q10) & (g.wspd <= g.q90))
                                  .mean()), 3)))
        for lh, g in pred.groupby("lead_h"):
            by_lead_rows.append(dict(
                config=cname, lead_h=int(lh), n=len(g),
                cov_q10=round(float((g.wspd < g.q10).mean()), 3),
                cov_q50=round(float((g.wspd < g.q50).mean()), 3),
                cov_q90=round(float((g.wspd < g.q90).mean()), 3),
                cov80=round(float(((g.wspd >= g.q10) & (g.wspd <= g.q90))
                                  .mean()), 3)))
        print(f"  {cname}: cov 0.1/0.5/0.9 = {cov10:.3f}/{cov50:.3f}/"
              f"{cov90:.3f}; 80%区间覆盖 {in80:.3f} 宽度 {width80:.2f}"
              f" ({time.time()-t0:.0f}s)", flush=True)

    pd.DataFrame(by_station_rows).to_csv(
        OUT / "loo_coverage_by_station.csv", index=False)
    pd.DataFrame(by_lead_rows).to_csv(OUT / "loo_coverage_by_lead.csv",
                                      index=False)
    (OUT / "loo_cov_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary["configs"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
