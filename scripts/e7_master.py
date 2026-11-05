#!/usr/bin/env python
"""Build intermediate keyed predictions, scores and diagnostic summaries.

Models include LightGBM quantiles, station MOS and Gaussian EMOS variants.
The script records sample flow, raw quantile crossings, reliability,
contingency counts and date-block resampling. E8 supplies the subsequent
distribution and scoring conventions; E10 supplies the three-model study.
Run ID: AWF_E7_MASTER_R1."""
import os
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
from e7_core import (block_bootstrap, cdf_at, crps_full, crps_normal_closed,
                     monotonize_rows, pinball, qs_k_rowwise)

DATA = Path(os.environ.get("AWF_DATA_ROOT", "data"))
PAIR = DATA / "derived" / "paired" / "paired_2021.parquet"
REMAF = Path("outputs/e5_baseline/rema/"
             "rema_features_by_station.csv")
OUT = Path("outputs/review_v2")
THR = 15.0
TAUS = np.round(np.concatenate([np.arange(0.05, 0.96, 0.05),
                                [0.97, 0.98, 0.99]]), 4)
SEED = 42
P = dict(n_estimators=400, learning_rate=0.05, num_leaves=31,
         min_data_in_leaf=50, subsample=0.9, subsample_freq=1,
         colsample_bytree=0.9, seed=SEED, num_threads=24,
         deterministic=True, force_col_wise=True, verbosity=-1)
FEATS = ["u10", "v10", "fcst_spd", "t2m", "prmsl", "rh2m", "lead_h",
         "month_sin", "month_cos", "lat", "lon", "elevation", "terrain"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = "AWF_E7_MASTER_R1"

    # ---------- A. 样本流程 ----------
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
    n_raw = len(df)
    n_miss = int(df.wspd.isna().sum())
    df = df.dropna(subset=["wspd"]).reset_index(drop=True)
    rema = pd.read_csv(REMAF)
    df = df.merge(rema, on="station", how="left")
    trn = df[df.month <= 8]
    va = df[df.month == 9]
    te = df[df.month >= 10].reset_index(drop=True)
    flow = dict(
        raw_paired_rows=n_raw, wspd_missing=n_miss, valid_rows=len(df),
        unique_station_valid=int(df.groupby(["station", "valid_time"]).ngroups),
        stations_year=int(df.station.nunique()),
        test_rows=len(te), test_stations=int(te.station.nunique()),
        train_rows=len(trn), val_rows=len(va),
        train_event_rate=round(float((trn.wspd > THR).mean()), 5),
        test_event_rate=round(float((te.wspd > THR).mean()), 5))
    (OUT / "sample_flow.json").write_text(
        json.dumps(flow, ensure_ascii=False, indent=2))
    print("[flow]", json.dumps(flow))
    y = te.wspd.values
    dates = pd.to_datetime(te.init).dt.date.values

    # ---------- B. 预测交付 ----------
    pred = te[["station", "init", "valid_time", "lead_h", "wspd",
               "fcst_spd"]].copy()
    pred = pred.rename(columns={"wspd": "obs", "fcst_spd": "gfs"})

    # B1: LGBM 22 分位
    Q = np.empty((len(te), len(TAUS)))
    q3 = {}
    for j, tau in enumerate(TAUS):
        m = lgb.LGBMRegressor(objective="quantile", alpha=float(tau), **P)
        m.fit(trn[FEATS], trn.wspd, categorical_feature=[FEATS.index("terrain")],
              eval_set=[(va[FEATS], va.wspd)], eval_metric="quantile",
              callbacks=[lgb.early_stopping(50, verbose=False)])
        Q[:, j] = m.predict(te[FEATS], num_iteration=m.best_iteration_)
        if abs(tau - 0.1) < 1e-9 or abs(tau - 0.5) < 1e-9 \
                or abs(tau - 0.9) < 1e-9:
            q3[round(float(tau), 2)] = Q[:, j]
        print(f"  lgbm tau={tau} fitted", flush=True)
    Qm = monotonize_rows(Q)
    for j, tau in enumerate(TAUS):
        pred[f"lgbm_q{int(round(tau*100)):02d}"] = Qm[:, j]

    # B2: stnMOS (站×lead 线性; 站不足20行回退逐站, 再回退池化)
    smos = np.full(len(te), np.nan)
    for s in te.station.unique():
        gs = trn[trn.station == s]
        for lh in te.lead_h.unique():
            g = gs[gs.lead_h == lh]
            if len(g) < 20:
                g = gs
            if len(g) < 10:
                continue
            b, a = np.polyfit(g.fcst_spd, g.wspd, 1)
            m = (te.station == s) & (te.lead_h == lh)
            smos[m] = a + b * te.fcst_spd[m]
    pred["stnmos"] = smos

    # B3: EMOS 高斯(闭式概率与CRPS同源)
    emos_q = {t: np.full(len(te), np.nan) for t in (0.1, 0.5, 0.9)}
    emos_mu = np.full(len(te), np.nan)
    emos_sd = np.full(len(te), np.nan)
    for lh in sorted(te.lead_h.unique()):
        trl = trn[trn.lead_h == lh]

        def obj(p):
            a, b, c, d = p
            if c <= 0:
                return 1e9
            return float(np.mean(crps_normal_closed(
                trl.wspd.values, a + b * trl.fcst_spd.values,
                np.maximum(c + d * trl.fcst_spd.values, 1e-3))))
        best, bv = None, np.inf
        for m0 in ([trl.wspd.mean(), 1.0, trl.wspd.std(), 0.0],
                   [0, 1, trl.wspd.std(), 0.05]):
            r = minimize(obj, m0, method="Nelder-Mead",
                         options=dict(maxiter=5000, xatol=1e-5, fatol=1e-6))
            if r.fun < bv:
                bv, best = r.fun, r.x
        m_ = te.lead_h == lh
        a, b, c, d = best
        emos_mu[m_] = a + b * te.fcst_spd[m_]
        emos_sd[m_] = np.maximum(c + d * te.fcst_spd[m_], 1e-3)
        for t in emos_q:
            emos_q[t][m_] = emos_mu[m_] + norm.ppf(t) * emos_sd[m_]
    pred["emos_mu"], pred["emos_sd"] = emos_mu, emos_sd

    # B4: EMOS-geo (信息可比概率基线: mu 依赖 elev+slope5k; 6参数)
    eg_mu = np.full(len(te), np.nan)
    eg_sd = np.full(len(te), np.nan)
    for lh in sorted(te.lead_h.unique()):
        trl = trn[trn.lead_h == lh]
        Xe = trl.elevation.values / 1000.0
        Xs = np.nan_to_num(trl.slope5k.values, nan=np.nanmedian(
            trn.slope5k.values)) / 10.0

        def objg(p):
            a, b, c1, c2, c3, dd = p
            if c3 <= 0:
                return 1e9
            mu = a + b * trl.fcst_spd.values + c1 * Xe + c2 * Xs
            sd = np.maximum(c3 + dd * trl.fcst_spd.values, 1e-3)
            return float(np.mean(crps_normal_closed(trl.wspd.values, mu, sd)))
        best, bv = None, np.inf
        for m0 in ([trl.wspd.mean(), 1.0, 0.0, 0.0, trl.wspd.std(), 0.0],):
            r = minimize(objg, m0, method="Nelder-Mead",
                         options=dict(maxiter=8000, maxfev=8000,
                                      xatol=1e-5, fatol=1e-6))
            if r.fun < bv:
                bv, best = r.fun, r.x
        m_ = te.lead_h == lh
        a, b, c1, c2, c3, dd = best
        eg_mu[m_] = (a + b * te.fcst_spd[m_] + c1 * te.elevation[m_] / 1000.0
                     + c2 * np.nan_to_num(te.slope5k[m_],
                                          nan=float(np.nanmedian(
                                              te.slope5k))) / 10.0)
        eg_sd[m_] = np.maximum(c3 + dd * te.fcst_spd[m_], 1e-3)
    pred["emosg_mu"], pred["emosg_sd"] = eg_mu, eg_sd
    pred["config"] = "E7_MASTER_R1"
    pred.to_parquet(OUT / "predictions_full.parquet", index=False)

    # ---------- C. 评分 ----------
    idx3 = [list(TAUS).index(round(t, 4))
            for t in (0.1, 0.5, 0.9)]
    Q3 = Qm[:, idx3]
    T3 = np.array([0.1, 0.5, 0.9])
    crps_lgbm_full, row_lgbm_full = crps_full(y, Qm, TAUS)
    crps_emos_full = float(np.mean(crps_normal_closed(y, emos_mu, emos_sd)))
    row_emos_full = crps_normal_closed(y, emos_mu, emos_sd)
    crps_eg_full = float(np.mean(crps_normal_closed(y, eg_mu, eg_sd)))
    row_eg_full = crps_normal_closed(y, eg_mu, eg_sd)
    qs22 = float(np.mean(qs_k_rowwise(y, Qm, TAUS)))
    # 交叉(原始精度)
    cross3 = (np.diff(Q3, axis=1) < -1e-12).any(axis=1)
    cross22 = (np.diff(Qm, axis=1) < -1e-12).any(axis=1)
    viol22 = np.maximum(-(np.diff(Qm, axis=1)), 0)
    # Brier: 训练期气候基准
    p_lgbm = 1 - cdf_at(Qm, TAUS, THR)
    p_emos = 1 - norm.cdf((THR - emos_mu) / emos_sd)
    p_eg = 1 - norm.cdf((THR - eg_mu) / eg_sd)
    p_clim_train = np.full(len(y), flow["train_event_rate"])
    ev = (y > THR).astype(float)
    brier = {k: float(np.mean((p - ev) ** 2)) for k, p in
             [("LGBM", p_lgbm), ("EMOS", p_emos), ("EMOSgeo", p_eg),
              ("clim_train", p_clim_train)]}
    # 可靠度(含 mean_pred)
    rel_rows = []
    for name, p in [("LGBM", p_lgbm), ("EMOS", p_emos), ("EMOSgeo", p_eg)]:
        bins = pd.cut(p, [0, .02, .05, .1, .2, .4, .8, 1.0])
        r = pd.DataFrame(dict(b=bins, p=p, e=ev)).groupby("b",
                                                          observed=True)
        for b_, g in r:
            rel_rows.append(dict(method=name, bin=str(b_), n=len(g),
                                 mean_pred=round(float(g.p.mean()), 4),
                                 obs_freq=round(float(g.e.mean()), 4)))
    pd.DataFrame(rel_rows).to_csv(OUT / "reliability_v2.csv", index=False)

    scores = dict(
        QS3_LGBM=round(float(np.mean(qs_k_rowwise(y, Q3, T3))), 4),
        QS3_EMOS=round(float(np.mean(
            [np.mean(pinball(y, emos_q[t], t)) for t in T3])) * 2 / 3, 4),
        QS22_LGBM=round(qs22, 4),
        CRPS_full_LGBM=round(crps_lgbm_full, 4),
        CRPS_full_EMOS=round(crps_emos_full, 4),
        CRPS_full_EMOSgeo=round(crps_eg_full, 4),
        MAE_LGBM=round(float(np.abs(q3[0.5] - y).mean()), 4),
        MAE_stnMOS=round(float(np.abs(smos - y).mean()), 4),
        MAE_stnMOS_n=int(np.isfinite(smos).sum()),
        brier=brier, crossing=dict(
            q3_rows=int(cross3.sum()),
            q22_rows=int(cross22.sum()), q22_rate=round(
                float(cross22.mean()), 4),
            q22_max_violation=round(float(viol22.max()), 4),
            q22_p99_violation=round(float(np.percentile(viol22, 99)), 4)))
    (OUT / "scores_summary.json").write_text(json.dumps(
        dict(scores=scores, taus=list(map(float, TAUS)),
             protocol=("QS_k=2/K*mean pinball (命名有限分位得分); 完整CRPS="
                       "2*梯形积分0..1, Q(tau)线性插值, 尾部dq外推(见e7_core"
                       "自检: Gaussian闭式vs数值一致, 退化==MAE)")),
        ensure_ascii=False, indent=2))
    print("[scores]", json.dumps(scores, ensure_ascii=False))

    # ---------- D. 修复版 bootstrap ----------
    ok = np.isfinite(smos)
    diffs = {
        "MAE_LGBM-stnMOS": (np.abs(q3[0.5] - y) - np.abs(smos - y))[ok],
        "MAE_LGBM-EMOS": np.abs(q3[0.5] - y) - np.abs(emos_mu - y),
        "QS3_LGBM-EMOS": 2.0 / 3.0 * (
            pinball(y, Q3[:, 0], 0.1) + pinball(y, Q3[:, 1], 0.5)
            + pinball(y, Q3[:, 2], 0.9)
            - (pinball(y, emos_q[0.1], 0.1) + pinball(y, emos_q[0.5], 0.5)
               + pinball(y, emos_q[0.9], 0.9))),
        "CRPSfull_LGBM-EMOS": row_lgbm_full - row_emos_full,
        "CRPSfull_LGBM-EMOSgeo": row_lgbm_full - row_eg_full,
        "Brier_LGBM-EMOS": (p_lgbm - ev) ** 2 - (p_emos - ev) ** 2,
    }
    boot_rows = []
    for name, dvec in diffs.items():
        dts = dates[ok] if name.startswith("MAE_LGBM-stnMOS") else dates
        for bl in (1, 3, 7):
            st = block_bootstrap(dvec, dts, bl, n_boot=1000, seed=SEED)
            lo, hi = np.percentile(st, [2.5, 97.5])
            boot_rows.append(dict(metric=name, block_days=bl,
                                  mean_diff=round(float(dvec.mean()), 4),
                                  ci_lo=round(float(lo), 4),
                                  ci_hi=round(float(hi), 4),
                                  excl0=bool(lo > 0 or hi < 0)))
    pd.DataFrame(boot_rows).to_csv(OUT / "bootstrap_v2.csv", index=False)
    print(pd.DataFrame(boot_rows).to_string(index=False))

    # ---------- E. 列联表(修复) ----------
    event = y > THR
    warn_v = {"GFS_thr": te.fcst_spd.values > THR,
              "QM_thr": None, "LGBM_q90": Qm[:, list(TAUS).index(0.9)] > THR,
              "EMOS_q90": emos_q[0.9] > THR}

    def cont(warn, ev_, n):
        H = int((warn & ev_).sum()); M = int((~warn & ev_).sum())
        F = int((warn & ~ev_).sum()); C = int((~warn & ~ev_).sum())
        return H, M, F, C, n

    rows = []
    for name, w in warn_v.items():
        if w is None:
            continue
        H, M, F, C, n = cont(w, event, len(y))
        rows.append(dict(method=name, lead_h="all", n=n, H=H, M=M, F=F, C=C,
                         POD=round(H / max(H + M, 1), 3),
                         POFD=round(F / max(F + C, 1), 3),
                         FAR=round(F / max(F + H, 1), 3),
                         CSI=round(H / max(H + M + F, 1), 3)))
        for lh in sorted(te.lead_h.unique()):
            m = te.lead_h.values == lh        # Select the lead-specific rows before evaluating counts.
            Hl, Ml, Fl, Cl, nl = cont(w[m], event[m], int(m.sum()))
            rows.append(dict(method=name, lead_h=int(lh), n=nl, H=Hl, M=Ml,
                             F=Fl, C=Cl,
                             POD=round(Hl / max(Hl + Ml, 1), 3),
                             POFD=round(Fl / max(Fl + Cl, 1), 3),
                             FAR=round(Fl / max(Fl + Hl, 1), 3),
                             CSI=round(Hl / max(Hl + Ml + Fl, 1), 3)))
    cv = pd.DataFrame(rows)
    # 不变量校验
    inv_ok = True
    for name in cv.method.unique():
        sub = cv[(cv.method == name) & (cv.lead_h != "all")]
        allr = cv[(cv.method == name) & (cv.lead_h == "all")].iloc[0]
        for k in ("H", "M", "F", "C"):
            if int(sub[k].sum()) != int(allr[k]):
                inv_ok = False
        if not (cv[cv.method == name].eval("H+M+F+C == n")).all():
            inv_ok = False
    cv.to_csv(OUT / "contingency_v2.csv", index=False)
    print("[contingency invariants]", "PASS" if inv_ok else "FAIL")
    print(cv[cv.lead_h == "all"].to_string(index=False))

    (OUT / "run_meta.json").write_text(json.dumps(dict(
        run_id=run_id, utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()
                                     ).hexdigest(),
        core_sha256=hashlib.sha256(
            (Path(__file__).parent / "e7_core.py").read_bytes()).hexdigest(),
        seed=SEED, env="conda dm"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
