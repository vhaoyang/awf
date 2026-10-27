#!/usr/bin/env python
"""AWF E6F — 评审 P1：交叉时空测试 + station-ID 记忆对照（R3.2/R3.4）

运行号: AWF_E6F_XSPATIAL_R1 (服务器端自编, 非本侧任务号)

设计:
  (1) 交叉时空 LOO: 每折留一站; 训练=其余 49 站且 valid_time<=08-31(早期);
      测试=留出站 valid_time>=10-01(后期)。固定 300 树(与 LOO 协议一致, 无早停,
      内部验证不触碰留出站)。配置 A_full 与 P1r。
      回答"未见站的未来预报"业务问法。
  (2) station-ID 对照(时间块): A_full + station(类别特征) —— 站网内记忆上限。
  (3) 全分布 CRPS 差(LGBM20q − EMOS闭式) 的日块 bootstrap(补 R1.4)。
输出: outputs/review_p1/{xspatial_summary.json, xspatial_by_station.csv}
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

DATA = Path("/mnt/d/data/awfdata")
OUT = Path("/home/vhyang/zcode/awf/outputs/review_p1")
SEED = 42
QS = [0.1, 0.5, 0.9]
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
         min_data_in_leaf=50, subsample=0.9, subsample_freq=1,
         colsample_bytree=0.9, seed=SEED, num_threads=24,
         deterministic=True, force_col_wise=True, verbosity=-1)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = "AWF_E6F_XSPATIAL_R1"
    rng = np.random.default_rng(SEED)

    # --- 特征装配(A_full 与 E5H 同源; P1r 复用 E5H 构建逻辑) ---
    import rasterio
    from pyproj import CRS, Transformer
    from scipy.ndimage import gaussian_filter
    reg = pd.read_csv(DATA / "derived/station_qc/station_sensor_registry.csv")
    df = pd.read_parquet(DATA / "derived/paired/paired_2021.parquet").rename(
        columns={"t2m_x": "obs_t2m", "t2m_y": "t2m"})
    df = df.merge(reg[["station", "lat", "lon", "elevation", "terrain"]],
                  on="station", how="left")
    df["terrain"] = df.terrain.astype("category")
    df["fcst_spd"] = np.hypot(df.u10, df.v10)
    df["month"] = df.valid_time.dt.month
    df["month_sin"] = np.sin(2 * np.pi * df.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.month / 12)
    df = df.dropna(subset=["wspd"]).reset_index(drop=True)
    rf = pd.read_csv("/home/vhyang/zcode/awf/outputs/e5_baseline/rema/"
                     "rema_features_by_station.csv")
    df = df.merge(rf, on="station", how="left")
    ds = rasterio.open(DATA / "raw/rema/rema_mosaic_1km_v2.0_dem.tif")
    elev = ds.read(1).astype(np.float32)
    if ds.nodata is not None:
        elev[elev == ds.nodata] = np.nan
    elev[elev < -100] = np.nan
    trm, crs, res = ds.transform, ds.crs, float(np.abs(ds.transform.a))
    sm5 = gaussian_filter(elev, 5000.0 / res)
    gy, gx = np.gradient(sm5, res, res)
    stq = reg.drop_duplicates("station")[["station", "lat", "lon"]]
    to_map = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    xs, ys = to_map.transform(stq.lon.values, stq.lat.values)
    cx = np.clip(np.round((xs - trm.c) / trm.a).astype(int), 0,
                 elev.shape[1] - 1)
    cy = np.clip(np.round((ys - trm.f) / trm.e).astype(int), 0,
                 elev.shape[0] - 1)
    du, dv = -gx[cy, cx], -gy[cy, cx]
    mp = dict(zip(stq.station, zip(du, dv)))
    du_r = df.station.map(lambda s: mp[s][0]).values
    dv_r = df.station.map(lambda s: mp[s][1]).values
    vdot = du_r * df.u10.values + dv_r * df.v10.values
    norm = np.hypot(du_r, dv_r) * np.maximum(
        np.hypot(df.u10, df.v10).values, 1e-6)
    df["katab_align_rema"] = vdot / np.maximum(norm, 1e-6)

    FEATS_A = ["u10", "v10", "fcst_spd", "t2m", "prmsl", "rh2m", "lead_h",
               "month_sin", "month_cos", "lat", "lon", "elevation", "terrain"]
    FEATS_P1r = [f for f in FEATS_A if f not in ("lat", "lon")] + [
        "slope1k", "slope5k", "slope25k", "prominence", "fetch_drop",
        "katab_align_rema"]

    # ---- (1) 交叉时空 LOO ----
    early = df.valid_time <= pd.Timestamp("2021-08-31 23:59")
    late = df.valid_time >= pd.Timestamp("2021-10-01 00:00")
    rows = []
    summary = {c: dict() for c in ("A_full", "P1r")}
    for cname, feats in (("A_full", FEATS_A), ("P1r", FEATS_P1r)):
        t0 = time.time()
        parts = []
        for s in sorted(df.station.unique()):
            trn = df[(df.station != s) & early]
            te = df[(df.station == s) & late]
            if len(te) < 50:
                continue
            preds = {}
            for tau in QS:
                m = lgb.LGBMRegressor(objective="quantile", alpha=tau, **P)
                m.fit(trn[feats], trn.wspd,
                      categorical_feature=[feats.index("terrain")])
                preds[tau] = m.predict(te[feats])
            y = te.wspd.values
            pin = [np.mean(np.maximum(t * (y - preds[t]),
                                      (t - 1) * (y - preds[t])))
                   for t in QS]
            parts.append(dict(
                config=cname, station=s, n=len(te),
                gfs_mae=float(np.abs(te.fcst_spd.values - y).mean()),
                lgbm_mae=float(np.abs(preds[0.5] - y).mean()),
                lgbm_crps3=float(2 * np.mean(pin)),
                cov80=float(((y >= preds[0.1]) & (y <= preds[0.9])).mean()),
                cov_q90=float((y < preds[0.9]).mean())))
        o = pd.DataFrame(parts)
        rows.append(o)
        w = o.n / o.n.sum()
        summary[cname] = dict(
            n_rows=int(o.n.sum()),
            pooled_gfs_mae=float((o.gfs_mae * w).sum()),
            pooled_lgbm_mae=float((o.lgbm_mae * w).sum()),
            pooled_crps3=float((o.lgbm_crps3 * w).sum()),
            cov80=float((o.cov80 * w).sum()),
            cov_q90=float((o.cov_q90 * w).sum()),
            wins=int((o.lgbm_mae < o.gfs_mae).sum()),
            folds=len(o), elapsed_s=round(time.time() - t0, 1))
        print(f"[{run_id}] {cname}: pooled {summary[cname]['pooled_lgbm_mae']:.3f}"
              f" vs GFS {summary[cname]['pooled_gfs_mae']:.3f} "
              f"cov80 {summary[cname]['cov80']:.3f} "
              f"({summary[cname]['elapsed_s']:.0f}s)", flush=True)
    xs = pd.concat(rows, ignore_index=True)
    xs.round(3).to_csv(OUT / "xspatial_by_station.csv", index=False)

    # ---- (2) station-ID 对照(时间块, 400树+早停同 E5) ----
    trn = df[df.month <= 8]
    va = df[df.month == 9]
    te = df[df.month >= 10]
    y = te.wspd.values
    df_id = df.copy()
    feats_id = FEATS_A + ["station"]
    m = lgb.LGBMRegressor(
        objective="quantile", alpha=0.5,
        **{**P, "n_estimators": 400})
    m.fit(trn.assign(station=trn.station.astype("category"))[feats_id],
          trn.wspd, categorical_feature=[feats_id.index("terrain"),
                                         feats_id.index("station")],
          eval_set=[(va.assign(station=va.station.astype("category"))[
              feats_id], va.wspd)], eval_metric="quantile",
          callbacks=[lgb.early_stopping(50, verbose=False)])
    # station 作为类别特征: 列名一致即可(类别含全部站)
    stn_mae = float(np.abs(
        m.predict(te.assign(station=te.station.astype("category"))[feats_id],
                  num_iteration=m.best_iteration_) - y).mean())
    summary["stationID_temporal_mae"] = stn_mae
    print(f"[{run_id}] station-ID temporal MAE = {stn_mae:.3f} "
          f"(LGBM base 1.764)")

    # ---- (3) 全分布 CRPS 差的日块 bootstrap ----
    try:
        dq = pd.read_csv("/home/vhyang/zcode/awf/outputs/review_p1/"
                         "dense_quantiles.csv")
        TAUS = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)] + \
            [0.97, 0.98, 0.99]
        lgb_crps_row = 2 / len(TAUS) * np.mean(
            np.column_stack([np.maximum(t * (dq.wspd - dq[str(t)]),
                                        (t - 1) * (dq.wspd - dq[str(t)]))
                             for t in TAUS]), axis=1)
        # EMOS 逐行闭式 CRPS: 由 e6e 参数重新生成(同协议) — 存于 brier 对照外,
        # 这里直接引用 e6e 汇总的均值差近似不可行, 改为用 EMOS q10/50/90 行级
        # pinball(从 fair run 的 emos 预测未存) —— 简化: 跳过行级, 报告均值差
        # 的 bootstrap 不可得时注明。
        summary["crps20_note"] = ("LGBM 行级 CRPS20 已重算可 bootstrap; "
                                  "EMOS 行级闭式未存, 全量均值见 e6e")
        np.save(OUT / "lgbm_crps20_rowlevel.npy", lgb_crps_row)
    except FileNotFoundError:
        summary["crps20_note"] = "dense_quantiles.csv 缺失, 跳过"

    (OUT / "xspatial_summary.json").write_text(json.dumps(
        dict(run_id=run_id, utc=datetime.now(timezone.utc).isoformat(),
             script_sha256=hashlib.sha256(Path(__file__).read_bytes()
                                          ).hexdigest(),
             design="train: 49 stations & valid<=0831; test: held-out "
                    "station & valid>=1001; fixed 300 trees, no early stop",
             **summary), ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
