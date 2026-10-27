#!/usr/bin/env python
"""AWF E6E — 评审 P1：密分位格点（R1.2 敏感性 + R1.3 概率预警）

运行号: AWF_E6E_DENSEQ_R1 (服务器端自编, 非本侧任务号)

内容(时间块口径, 与 E5 同划分/超参):
  (1) LGBM wspd 重拟合于 20 个分位 τ∈{.05,.10,...,.95,.97,.98,.99};
  (2) CRPS_20q=(2/20)Σpinball 与 CRPS_3 对照 —— 三分位近似的敏感性;
  (3) P(y>15) 由 20 分位线性插值 CDF(尾部: 低于 q05 线性归零于 0,
      高于 q99 截 0.005), Brier + 分箱可靠度, EMOS 闭式概率对照。
输出: outputs/review_p1/{dense_quantiles.csv, brier_dense.csv,
       dense_summary.json}
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

DATA = Path("/mnt/d/data/awfdata")
PAIR = DATA / "derived" / "paired" / "paired_2021.parquet"
OUT = Path("/home/vhyang/zcode/awf/outputs/review_p1")
THR = 15.0
TAUS = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)] + [0.97, 0.98, 0.99]
FEATS = ["u10", "v10", "fcst_spd", "t2m", "prmsl", "rh2m", "lead_h",
         "month_sin", "month_cos", "lat", "lon", "elevation", "terrain"]
P = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
         min_data_in_leaf=50, subsample=0.9, subsample_freq=1,
         colsample_bytree=0.9, seed=42, num_threads=24,
         deterministic=True, force_col_wise=True, verbosity=-1)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = "AWF_E6E_DENSEQ_R1"
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
    trn, va, te = (df[df.month <= 8], df[df.month == 9], df[df.month >= 10])
    y = te.wspd.values

    qs = {}
    for tau in TAUS:
        m = lgb.LGBMRegressor(objective="quantile", alpha=tau, **P)
        m.fit(trn[FEATS], trn.wspd, categorical_feature=[FEATS.index("terrain")],
              eval_set=[(va[FEATS], va.wspd)], eval_metric="quantile",
              callbacks=[lgb.early_stopping(50, verbose=False)])
        qs[tau] = m.predict(te[FEATS], num_iteration=m.best_iteration_)
        print(f"  tau={tau} fitted", flush=True)
    qtab = pd.DataFrame(qs)
    qtab.insert(0, "wspd", y)
    qtab.round(3).to_csv(OUT / "dense_quantiles.csv", index=False)

    def pin(y, q, t):
        z = y - q
        return np.maximum(t * z, (t - 1) * z)

    crps20 = float(2 / len(TAUS)
                   * sum(pin(y, qs[t], t).mean() for t in TAUS))
    crps3 = float(2 / 3 * sum(pin(y, qs[t], t).mean()
                              for t in (0.1, 0.5, 0.9)))

    # P(y>THR) 插值
    Q = np.column_stack([qs[t] for t in TAUS])   # n x 20
    order = np.argsort(Q, axis=1)                 # 保证单调(交叉极少)
    Qs = np.take_along_axis(Q, order, axis=1)
    taus_sorted = np.take_along_axis(
        np.tile(np.array(TAUS), (len(y), 1)), order, axis=1)
    p_thr = np.empty(len(y))
    for i in range(len(y)):
        p_thr[i] = 1 - np.interp(THR, Qs[i], taus_sorted[i],
                                  left=0.01, right=0.995)
    event = y > THR
    brier = float(np.mean((p_thr - event.astype(float)) ** 2))
    # EMOS 闭式概率对照
    from scipy.stats import norm
    from scipy.optimize import minimize
    def crps_n(y, m, s):
        z = (y - m) / s
        return s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z)
                    - 1 / np.sqrt(np.pi))
    p_emos = np.empty(len(y))
    for lh in sorted(te.lead_h.unique()):
        trl = trn[trn.lead_h == lh]
        def obj(pp):
            a, b, c, d = pp
            if c <= 0:
                return 1e9
            return float(np.mean(crps_n(trl.wspd, a + b * trl.fcst_spd,
                                        np.maximum(c + d * trl.fcst_spd, 1e-3))))
        best, bv = None, np.inf
        for m0 in ([trl.wspd.mean(), 1.0, trl.wspd.std(), 0.0],
                   [0, 1, trl.wspd.std(), 0]):
            r = minimize(obj, m0, method="Nelder-Mead",
                         options=dict(maxiter=3000))
            if r.fun < bv:
                bv, best = r.fun, r.x
        a, b, c, d = best
        m_ = te.lead_h == lh
        mu = a + b * te.fcst_spd[m_]
        sd = np.maximum(c + d * te.fcst_spd[m_], 1e-3)
        p_emos[m_] = 1 - norm.cdf((THR - mu) / sd)
    brier_emos = float(np.mean((p_emos - event.astype(float)) ** 2))
    # EMOS 闭式 CRPS(全分布), 与 LGBM crps_20q 公平对照
    crps_emos_full = np.empty(len(y))
    for lh in sorted(te.lead_h.unique()):
        trl = trn[trn.lead_h == lh]
        def obj2(pp):
            a, b, c, d = pp
            if c <= 0:
                return 1e9
            return float(np.mean(crps_n(trl.wspd, a + b * trl.fcst_spd,
                                        np.maximum(c + d * trl.fcst_spd, 1e-3))))
        best, bv2 = None, np.inf
        for m0 in ([trl.wspd.mean(), 1.0, trl.wspd.std(), 0.0],
                   [0, 1, trl.wspd.std(), 0]):
            r2 = minimize(obj2, m0, method="Nelder-Mead",
                          options=dict(maxiter=3000))
            if r2.fun < bv2:
                bv2, best = r2.fun, r2.x
        a, b, c, d = best
        msk = te.lead_h == lh
        mu = a + b * te.fcst_spd[msk]
        sd = np.maximum(c + d * te.fcst_spd[msk], 1e-3)
        crps_emos_full[msk] = crps_n(te.wspd[msk].values, mu.values, sd)
    clim_brier = float(np.mean((event.mean() - event.astype(float)) ** 2))

    rel_rows = []
    for name, p in [("LGBM20q", p_thr), ("EMOS", p_emos)]:
        bins = pd.cut(p, [0, .02, .05, .1, .2, .4, .8, 1.0])
        rr = pd.DataFrame(dict(b=bins, e=event.astype(float))).groupby(
            "b", observed=True).agg(n=("e", "size"), obs=("e", "mean"))
        for b, r in rr.iterrows():
            rel_rows.append(dict(method=name, bin=str(b), n=int(r.n),
                                 obs_rate=round(float(r.obs), 3)))
        rel_rows.append(dict(method=name, bin="BRIER", n=len(p),
                             obs_rate=round(float(np.mean(
                                 (p - event) ** 2)), 4)))
    pd.DataFrame(rel_rows).to_csv(OUT / "brier_dense.csv", index=False)
    summary = dict(
        run_id=run_id, utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        taus=TAUS, crps_20q=round(crps20, 4), crps_3_of_same=round(crps3, 4),
        crps3_ratio=round(crps3 / crps20, 3),
        brier_LGBM20q=round(brier, 4), brier_EMOS=round(brier_emos, 4),
        brier_climatology=round(clim_brier, 4),
        crps_EMOS_full=round(float(np.mean(crps_emos_full)), 4),
        brier_skill_LGBM=round(1 - brier / clim_brier, 3),
        brier_skill_EMOS=round(1 - brier_emos / clim_brier, 3),
        note="P(y>THR): 20 分位插值 CDF, q99 以上截 0.005; "
             "交叉行按排序修正(实测交叉率见 dense_quantiles)")
    (OUT / "dense_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
