#!/usr/bin/env python
"""Compare feature subsets in the early LightGBM baseline.

The configurations and temporal split are defined below. This experiment
precedes the three-model, six-design E10 comparison.
Run ID: AWF_E5D_ABL_R1."""
import os
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

DATA = Path(os.environ.get("AWF_DATA_ROOT", "data"))
PAIR = DATA / "derived" / "paired" / "paired_2021.parquet"
REG = DATA / "derived" / "station_qc" / "station_sensor_registry.csv"
OUT = Path("outputs/e5_baseline/ablation")

SEED = 42
THREADS = 24
QUANTILES = [0.1, 0.5, 0.9]
P = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
         min_data_in_leaf=50, subsample=0.9, subsample_freq=1,
         colsample_bytree=0.9, seed=SEED, num_threads=THREADS,
         deterministic=True, force_col_wise=True, verbosity=-1)
ALLF = ["u10", "v10", "fcst_spd", "t2m", "prmsl", "rh2m", "lead_h",
        "month_sin", "month_cos", "lat", "lon", "elevation", "terrain"]
GROUPS = {
    "full": ALLF,
    "no_terrain": [f for f in ALLF if f != "terrain"],
    "no_geo": [f for f in ALLF if f not in ("lat", "lon", "elevation", "terrain")],
    "no_lead": [f for f in ALLF if f != "lead_h"],
    "no_season": [f for f in ALLF if f not in ("month_sin", "month_cos")],
    "gfs_only": ["u10", "v10", "fcst_spd", "t2m", "prmsl", "rh2m"],
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = "AWF_E5D_ABL_R1"
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
    df = df.dropna(subset=["wspd"])

    tr = df[df.month <= 8]
    va = df[df.month == 9]
    te = df[df.month >= 10]
    y = te.wspd.values

    rows = []
    for gname, feats in GROUPS.items():
        cat = [i for i, f in enumerate(feats) if f == "terrain"]
        qs = {}
        for tau in QUANTILES:
            m = lgb.LGBMRegressor(objective="quantile", alpha=tau, **P)
            m.fit(tr[feats], tr.wspd, categorical_feature=cat or None,
                  eval_set=[(va[feats], va.wspd)], eval_metric="quantile",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            qs[tau] = m.predict(te[feats],
                                num_iteration=m.best_iteration_)
        pin = []
        for tau in QUANTILES:
            z = y - qs[tau]
            pin.append(float(np.mean(np.maximum(tau * z, (tau - 1) * z))))
        rows.append(dict(
            group=gname, n_feats=len(feats),
            mae=float(np.abs(qs[0.5] - y).mean()),
            crps3q=float(2 * np.mean(pin)),
        ))
        print(f"[{run_id}] {gname}: mae={rows[-1]['mae']:.3f} "
              f"crps3q={rows[-1]['crps3q']:.3f}", flush=True)

    out = pd.DataFrame(rows)
    base = out[out.group == "full"].iloc[0]
    out["d_mae_pct"] = (100 * (out.mae - base.mae) / base.mae).round(2)
    out["d_crps_pct"] = (100 * (out.crps3q - base.crps3q) / base.crps3q).round(2)
    out.round(4).to_csv(OUT / "ablation.csv", index=False)
    (OUT / "abl_summary.json").write_text(json.dumps(dict(
        run_id=run_id, utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        groups=GROUPS, rows=out.round(4).to_dict("records"),
        denominators="测试集站点-小时(与E5一致 n=%d)" % len(te)),
        ensure_ascii=False, indent=2))
    print(out.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
