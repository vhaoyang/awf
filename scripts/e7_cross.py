#!/usr/bin/env python
"""AWF E7 — 原始(排序前)分位交叉统计: 确定性重拟合22分位, 报告交叉率/幅度
运行号: AWF_E7_CROSS_R1。与 E7_MASTER 同种子同协议 => 模型逐位一致。"""
import sys, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
DATA = Path("/mnt/d/data/awfdata")
OUT = Path("/home/vhyang/zcode/awf/outputs/review_v2")
TAUS = np.round(np.concatenate([np.arange(0.05,0.96,0.05),[0.97,0.98,0.99]]),4)
P = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
         min_data_in_leaf=50, subsample=0.9, subsample_freq=1,
         colsample_bytree=0.9, seed=42, num_threads=24,
         deterministic=True, force_col_wise=True, verbosity=-1)
FEATS = ["u10","v10","fcst_spd","t2m","prmsl","rh2m","lead_h",
         "month_sin","month_cos","lat","lon","elevation","terrain"]
df = pd.read_parquet(DATA/"derived/paired/paired_2021.parquet").rename(
    columns={"t2m_x":"obs_t2m","t2m_y":"t2m"})
reg = pd.read_csv(DATA/"derived/station_qc/station_sensor_registry.csv")
df = df.merge(reg[["station","lat","lon","elevation","terrain"]], on="station", how="left")
df["terrain"]=df.terrain.astype("category"); df["fcst_spd"]=np.hypot(df.u10,df.v10)
df["month"]=df.valid_time.dt.month
df["month_sin"]=np.sin(2*np.pi*df.month/12); df["month_cos"]=np.cos(2*np.pi*df.month/12)
df=df.dropna(subset=["wspd"]).reset_index(drop=True)
trn,va,te = df[df.month<=8], df[df.month==9], df[df.month>=10]
Q = np.empty((len(te), len(TAUS)))
for j,tau in enumerate(TAUS):
    m = lgb.LGBMRegressor(objective="quantile", alpha=float(tau), **P)
    m.fit(trn[FEATS], trn.wspd, categorical_feature=[FEATS.index("terrain")],
          eval_set=[(va[FEATS],va.wspd)], eval_metric="quantile",
          callbacks=[lgb.early_stopping(50,verbose=False)])
    Q[:,j]=m.predict(te[FEATS], num_iteration=m.best_iteration_)
    print(f"tau={tau}", flush=True)
np.save(OUT/"raw_Q22.npy", Q)
d = np.diff(Q, axis=1)
viol = np.maximum(-d, 0)
rows_cross = (viol > 1e-12).any(axis=1)
big = viol > 0.5
near_thr = None
stats = dict(
    run_id="AWF_E7_CROSS_R1",
    utc=datetime.now(timezone.utc).isoformat(),
    n=len(te), rows_with_crossing=int(rows_cross.sum()),
    rate=round(float(rows_cross.mean()),4),
    max_violation=round(float(viol.max()),4),
    p99_violation=round(float(np.percentile(viol,99)),4),
    mean_violation_when_crossed=round(float(viol[viol>0].mean()) if (viol>0).any() else 0,4),
    rows_violation_gt_0p5=int(big.sum()),
    pairs_crossed=int((viol>1e-12).sum()),
    pairs_total=int(viol.size))
(OUT/"crossing_raw_stats.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2))
print(json.dumps(stats,ensure_ascii=False))
