#!/usr/bin/env python
"""AWF E8 — P1r_fix 前向未见站重跑（V3-04 修复后地形特征; 正式脚本替代临时内联版）

运行号: AWF_E8_FWD_R1 (服务器端自编, 非本侧任务号)

对应 V3-04/V3-05/V3-07(前向训练脚本归档):
  - 训练=站 s 之外的早期样本(valid_time<=2021-08-31), 测试=站 s 的
    10-12 月样本; 测试样本<50 的折跳过并逐站记录(实际 42 折)。
  - 地形特征与 e8_loo 完全同源(terrain_features_v3.csv)。

输出: outputs/review_v3/{fwd_e8_pred_P1r_fix.parquet,
       fwd_e8_summary_P1r_fix.json, fwd_e8_skipped.json}
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
from e8_core import monotonize_rows
from e8_loo import FEATS_A, REMA_STATIC, P, QS, attach_features

DATA = Path("/mnt/d/data/awfdata")
PAIR = DATA / "derived" / "paired" / "paired_2021.parquet"
REMAF = Path("/home/vhyang/zcode/awf/outputs/e5_baseline/rema/"
             "rema_features_by_station.csv")
OUT = Path("/home/vhyang/zcode/awf/outputs/review_v3")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = "AWF_E8_FWD_R1"
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
    early = df.valid_time <= pd.Timestamp("2021-08-31 23:59")
    late = df.valid_time >= pd.Timestamp("2021-10-01 00:00")

    t0 = time.time()
    parts, skipped = [], []
    for s in sorted(df.station.unique()):
        trn = df[(df.station != s) & early]
        te = df[(df.station == s) & late]
        if len(te) < 50:
            skipped.append(dict(station=s, n_test_rows=int(len(te))))
            continue
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
        print(f"  fwd fold {s}: n={len(p)} ({time.time()-t0:.0f}s)",
              flush=True)
    pred = pd.concat(parts, ignore_index=True)
    pred.to_parquet(OUT / "fwd_e8_pred_P1r_fix.parquet", index=False)
    (OUT / "fwd_e8_skipped.json").write_text(json.dumps(
        skipped, ensure_ascii=False, indent=2))
    y = pred.obs.values
    pin = [np.mean(np.maximum(t * (y - pred[f"q{int(t*100)}"]),
                              (t - 1) * (y - pred[f"q{int(t*100)}"])))
           for t in QS]
    summary = dict(
        run_id=run_id, utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        config="P1r_fix_v3terrain", n_rows=len(pred),
        folds=int(pred.station.nunique()), skipped=len(skipped),
        pooled_mae=round(float(abs(pred.q50 - y).mean()), 4),
        pooled_gfs_mae=round(float(abs(pred.gfs - y).mean()), 4),
        qs3=round(float(2 * np.mean(pin)), 4),
        cov80=round(float(((y >= pred.q10) & (y <= pred.q90)).mean()), 4),
        elapsed_s=round(time.time() - t0, 1),
        features=feats)
    (OUT / "fwd_e8_summary_P1r_fix.json").write_text(json.dumps(
        summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
