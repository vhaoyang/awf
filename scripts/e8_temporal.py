#!/usr/bin/env python
"""AWF E8 — 时间测试重跑（V3-05: 以同版本配置替换旧 1.725 支撑）

运行号: AWF_E8_TEMPORAL_R1 (服务器端自编, 非本侧任务号)

对应 V3-05: 旧 1.725 来自 E5G P2(旧对齐度/GFS-HGT 梯度), 不得继续
支撑修复后地形配置的"时间技能保持"。本脚本以与 e8_loo/e8_fwd 完全
同源的地形特征, 在标准时间划分(train<=8月, val=9月早停, test>=10月)
重跑 A_full 与 P1r_fix, 输出同口径 MAE/QS3/cov80。

输出: outputs/review_v3/temporal_e8_summary.json
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
from e8_loo import FEATS_A, REMA_STATIC, attach_features

DATA = Path("/mnt/d/data/awfdata")
PAIR = DATA / "derived" / "paired" / "paired_2021.parquet"
REMAF = Path("/home/vhyang/zcode/awf/outputs/e5_baseline/rema/"
             "rema_features_by_station.csv")
OUT = Path("/home/vhyang/zcode/awf/outputs/review_v3")
QS = [0.1, 0.5, 0.9]
P = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
         min_data_in_leaf=50, subsample=0.9, subsample_freq=1,
         colsample_bytree=0.9, seed=42, num_threads=24,
         deterministic=True, force_col_wise=True, verbosity=-1)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = "AWF_E8_TEMPORAL_R1"
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
    trn = df[df.month <= 8]
    va = df[df.month == 9]
    te = df[df.month >= 10].reset_index(drop=True)
    configs = {
        "A_full": FEATS_A,
        "P1r_fix": [f for f in FEATS_A if f not in ("lat", "lon")]
        + REMA_STATIC + ["katab_align_EN", "is_flat", "is_missing"],
    }
    out = dict(run_id=run_id,
               utc=datetime.now(timezone.utc).isoformat(),
               script_sha256=hashlib.sha256(
                   Path(__file__).read_bytes()).hexdigest(),
               split="train<=Aug, val=Sep(early stop), test>=Oct",
               configs={})
    y = te.wspd.values
    rema_sha = hashlib.sha256(Path(REMAF).read_bytes()).hexdigest()
    terr_sha = hashlib.sha256(
        Path("/home/vhyang/zcode/awf/outputs/review_v3/"
             "terrain_features_v3.csv").read_bytes()).hexdigest()
    for cname, feats in configs.items():
        t0 = time.time()
        Q = np.empty((len(te), len(QS)))
        Q_raw = np.empty((len(te), len(QS)))
        best_iters = []
        for j, tau in enumerate(QS):
            m = lgb.LGBMRegressor(objective="quantile", alpha=tau, **P)
            m.fit(trn[feats], trn.wspd,
                  categorical_feature=[feats.index("terrain")],
                  eval_set=[(va[feats], va.wspd)],
                  eval_metric="quantile",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            Q_raw[:, j] = m.predict(te[feats],
                                    num_iteration=m.best_iteration_)
            best_iters.append(int(m.best_iteration_ or P["n_estimators"]))
        Q = monotonize_rows(Q_raw)
        pin = [np.mean(np.maximum(t * (y - Q[:, j]),
                                  (t - 1) * (y - Q[:, j])))
               for j, t in enumerate(QS)]
        pred = te[["station", "init", "valid_time", "lead_h", "wspd",
                   "fcst_spd"]].copy().rename(columns={"wspd": "obs",
                                                       "fcst_spd": "gfs"})
        for j, t in enumerate(QS):
            k = int(t * 100)
            pred[f"q{k}"] = Q[:, j]              # 重排后(单调, 主口径)
            pred[f"q{k}_raw"] = Q_raw[:, j]      # 原始拟合(交叉核查)
        pred.to_parquet(OUT / f"temporal_e8_pred_{cname}.parquet",
                        index=False)
        out["configs"][cname] = dict(
            n_rows=len(te),
            mae=round(float(np.abs(Q[:, 1] - y).mean()), 4),
            gfs_mae=round(float(np.abs(te.fcst_spd.values - y).mean()), 4),
            qs3=round(float(2 * np.mean(pin)), 4),
            cov80=round(float(((y >= Q[:, 0]) & (y <= Q[:, 2])).mean()), 4),
            best_iterations=best_iters,
            model_registration=(
                "temporal three-quantile speed config; distinct from the "
                "22-quantile rearranged main experiment (Table 2)"),
            features=feats,
            rema_static_source=dict(path=str(REMAF), sha256=rema_sha),
            terrain_v3_source=dict(
                path="outputs/review_v3/terrain_features_v3.csv",
                sha256=terr_sha),
            seed=42, env="conda dm",
            elapsed_s=round(time.time() - t0, 1))
        print(f"[{cname}]", json.dumps(out["configs"][cname]), flush=True)
    (OUT / "temporal_e8_summary.json").write_text(json.dumps(
        out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
