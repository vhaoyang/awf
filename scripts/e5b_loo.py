#!/usr/bin/env python
"""Evaluate the early LightGBM baseline by leaving out one station at a time.

Folds use available paired observations, not a complete calendar year.
Direction and u/v summaries require trusted, nonmissing direction data.
Run ID: AWF_E5B_LOO_R1."""
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

DATA = Path(os.environ.get("AWF_DATA_ROOT", "data"))
PAIR = DATA / "derived" / "paired" / "paired_2021.parquet"
REG = DATA / "derived" / "station_qc" / "station_sensor_registry.csv"
OUT = Path("outputs/e5_baseline/loo")

SEED = 42
THREADS = 24
N_EST = 300
QUANTILES = [0.1, 0.5, 0.9]
P = dict(n_estimators=N_EST, learning_rate=0.05, num_leaves=31,
         min_data_in_leaf=50, subsample=0.9, subsample_freq=1,
         colsample_bytree=0.9, seed=SEED, num_threads=THREADS,
         deterministic=True, force_col_wise=True, verbosity=-1)
FEATS = ["u10", "v10", "fcst_spd", "t2m", "prmsl", "rh2m", "lead_h",
         "month_sin", "month_cos", "lat", "lon", "elevation", "terrain"]
CAT = FEATS.index("terrain")


def circ_deg(d):
    return (d + 180.0) % 360.0 - 180.0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = "AWF_E5B_LOO_R1"
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

    stations = sorted(df.station.unique())
    rows = []
    t0 = time.time()
    for k, s in enumerate(stations, 1):
        tr = df[df.station != s]
        te = df[df.station == s]
        preds = {}
        for tgt in ["wspd", "obs_u", "obs_v"]:
            taus = QUANTILES if tgt == "wspd" else [0.5]
            trt = tr[tr[tgt].notna()]
            for tau in taus:
                m = lgb.LGBMRegressor(
                    objective="quantile", alpha=tau, **P)
                m.fit(trt[FEATS], trt[tgt], categorical_feature=[CAT])
                preds[(tgt, tau)] = m.predict(te[FEATS])
        y = te.wspd.values
        e_g = te.fcst_spd.values - y
        e_l = preds[("wspd", 0.5)] - y
        pin = []
        for tau in QUANTILES:
            z = y - preds[("wspd", tau)]
            pin.append(float(np.mean(np.maximum(tau * z, (tau - 1) * z))))
        trust = te.dir_trust.values & te.obs_u.notna().values
        d_g = circ_deg(np.degrees(np.arctan2(-te.u10.values[trust],
                                             -te.v10.values[trust]))
                       - te.wdir.values[trust])
        lu, lv = preds[("obs_u", 0.5)], preds[("obs_v", 0.5)]
        m2 = trust & ~np.isnan(lu) & ~np.isnan(lv)
        d_l = circ_deg(np.degrees(np.arctan2(-lu[m2], -lv[m2]))
                       - te.wdir.values[m2])
        rows.append(dict(
            station=s, terrain=str(te.terrain.iloc[0]), n=len(te),
            n_trust=int(m2.sum()),
            gfs_mae=float(np.abs(e_g).mean()),
            lgbm_mae=float(np.abs(e_l).mean()),
            lgbm_crps3q=float(2 * np.mean(pin)),
            gfs_dir_mae=float(np.abs(d_g).mean()),
            lgbm_dir_mae=float(np.abs(d_l).mean()),
        ))
        if k % 5 == 0 or k == len(stations):
            print(f"[{run_id}] {k}/{len(stations)} stations "
                  f"({time.time() - t0:.0f}s)", flush=True)

    out = pd.DataFrame(rows)
    out["mae_gain_pct"] = (100 * (out.gfs_mae - out.lgbm_mae)
                           / out.gfs_mae).round(2)
    out.round(3).to_csv(OUT / "loo_by_station.csv", index=False)
    w = out.n / out.n.sum()
    summary = dict(
        run_id=run_id, utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        n_folds=len(out), n_rows=int(out.n.sum()),
        denominators=dict(station_equal="各站指标未加权平均(站等权)",
                          pooled="按行数加权(站点-小时)"),
        station_equal=dict(
            gfs_mae=float(out.gfs_mae.mean()), lgbm_mae=float(out.lgbm_mae.mean()),
            lgbm_crps3q=float(out.lgbm_crps3q.mean()),
            gfs_dir_mae=float(out.gfs_dir_mae.mean()),
            lgbm_dir_mae=float(out.lgbm_dir_mae.mean()),
            mae_gain_pct=float(out.mae_gain_pct.mean())),
        pooled=dict(
            gfs_mae=float((out.gfs_mae * w).sum()),
            lgbm_mae=float((out.lgbm_mae * w).sum()),
            lgbm_crps3q=float((out.lgbm_crps3q * w).sum())),
        n_stations_lgbm_wins=int((out.lgbm_mae < out.gfs_mae).sum()),
        worst5=out.nsmallest(5, "mae_gain_pct")[
            ["station", "terrain", "n", "mae_gain_pct"]].to_dict("records"),
        best5=out.nlargest(5, "mae_gain_pct")[
            ["station", "terrain", "n", "mae_gain_pct"]].to_dict("records"),
        elapsed_s=round(time.time() - t0, 1),
    )
    (OUT / "loo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
