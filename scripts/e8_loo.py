#!/usr/bin/env python
"""Fit the intermediate held-station terrain model and export keyed predictions.

Alignment and missing/flat flags use terrain_features_v3.csv. The historical
A_full comparison is read from the corresponding earlier output files.
Run ID: AWF_E8_LOO_R1."""
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

sys.path.insert(0, str(Path(__file__).parent))
from e8_core import monotonize_rows

DATA = Path(os.environ.get("AWF_DATA_ROOT", "data"))
PAIR = DATA / "derived" / "paired" / "paired_2021.parquet"
REMAF = Path("outputs/e5_baseline/rema/"
             "rema_features_by_station.csv")
TFV3 = Path("outputs/review_v3/terrain_features_v3.csv")
OUT = Path("outputs/review_v3")
QS = [0.1, 0.5, 0.9]
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=31,
         min_data_in_leaf=50, subsample=0.9, subsample_freq=1,
         colsample_bytree=0.9, seed=42, num_threads=24,
         deterministic=True, force_col_wise=True, verbosity=-1)
FEATS_A = ["u10", "v10", "fcst_spd", "t2m", "prmsl", "rh2m", "lead_h",
           "month_sin", "month_cos", "lat", "lon", "elevation", "terrain"]
REMA_STATIC = ["slope1k", "slope5k", "slope25k", "prominence", "fetch_drop"]


def attach_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach alignment and flags from the E8 terrain-feature table."""
    tf = pd.read_csv(TFV3).set_index("station")
    dE = df.station.map(tf.downhill_E)
    dN = df.station.map(tf.downhill_N)
    w = np.maximum(np.hypot(df.u10.values, df.v10.values), 1e-9)
    df["katab_align_EN"] = (dE.values * df.u10.values
                            + dN.values * df.v10.values) / w
    status = df.station.map(tf.status)
    df["is_flat"] = (status == "flat").astype(int).values
    df["is_missing"] = (status == "missing").astype(int).values
    # flat/missing 站不伪造方向(下坡向量本来即 0/NaN -> 对齐度置 0)
    df.loc[df.is_flat == 1, "katab_align_EN"] = 0.0
    df.loc[df.is_missing == 1, "katab_align_EN"] = 0.0
    return df


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = "AWF_E8_LOO_R1"
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
    df = attach_features(df)
    feats = [f for f in FEATS_A if f not in ("lat", "lon")] + \
        REMA_STATIC + ["katab_align_EN", "is_flat", "is_missing"]

    t0 = time.time()
    parts, st_rows = [], []
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
                "fcst_spd"]].copy().rename(columns={"wspd": "obs",
                                                    "fcst_spd": "gfs"})
        p["q10"], p["q50"], p["q90"] = Q[:, 0], Q[:, 1], Q[:, 2]
        parts.append(p)
        y = p.obs.values
        st_rows.append(dict(station=s, n=len(p),
                            mae=abs(p.q50 - y).mean(),
                            gfs_mae=abs(p.gfs - y).mean()))
        print(f"  fold {s}: mae {st_rows[-1]['mae']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    pred = pd.concat(parts, ignore_index=True)
    pred.to_parquet(OUT / "loo_e8_pred_P1r_fix.parquet", index=False)
    pd.DataFrame(st_rows).to_csv(OUT / "loo_e8_station_P1r_fix.csv",
                                 index=False)
    y = pred.obs.values
    pin = [np.mean(np.maximum(t * (y - pred[f"q{int(t*100)}"]),
                              (t - 1) * (y - pred[f"q{int(t*100)}"])))
           for t in QS]
    summary = dict(
        run_id=run_id, utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config="P1r_fix_v3terrain", n_rows=len(pred),
        pooled_mae=round(float(abs(pred.q50 - y).mean()), 4),
        pooled_gfs_mae=round(float(abs(pred.gfs - y).mean()), 4),
        qs3=round(float(2 * np.mean(pin)), 4),
        cov80=round(float(((y >= pred.q10) & (y <= pred.q90)).mean()), 4),
        elapsed_s=round(time.time() - t0, 1),
        features=feats,
        terrain_source=str(TFV3))
    (OUT / "loo_e8_summary.json").write_text(json.dumps(
        summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
