#!/usr/bin/env python
"""AWF E8 — V3 返修主重算：统一分布/精确CRPS/守恒可靠度/截断bootstrap/EMOS-geo填充修复

运行号: AWF_E8_SCORES_R1 (服务器端自编, 非本侧任务号)

对应 V3-01/02/03/06:
  - 输入冻结: review_v2/predictions_full.parquet(键+单调分位) 与 raw_Q22.npy
    (原始精度), 校验 raw 排序后与交付一致(键对齐)。
  - 唯一完整分布: e8_core.build_complete_Q(非负支撑, 下尾线性至 0,
    上尾延续末段斜率); CRPS/超越概率/分位诊断同源; 精确分段积分。
  - 回归核对: 旧端点规则(Q0=max(q05-dq,0),Q1=q99+dq)下精确 CRPS 应≈
    1.2464638(评审独立复算), 允差 1e-5。
  - 可靠度: include_lowest 零概率箱纳入, 逐方法 Σn==18235、事件数守恒,
    Wilson 95% CI。
  - bootstrap: 截断协议(抽中天数==唯一日期数), 输出每次重抽统计量。
  - EMOS-geo: 坡度缺测填充固定为训练期中位数(训练与预测同值), 交付
    拟合参数/收敛/受影响行数。

输出: outputs/review_v3/{predictions_v3.parquet, raw_Q22_keyed.parquet,
  scores_summary_v3.json, reliability_v3.csv, bootstrap_v3.csv,
  bootstrap_stats_v3.npz, contingency_v3.csv, threshold_curves_v3.csv,
  crossing_v3.json, sample_flow_v3.json, emosg_fit_v3.json, run_meta.json}
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
from e8_core import (block_bootstrap, build_complete_Q, crps_normal_closed,
                     exact_crps_rows, monotonize_rows, pinball, prob_le,
                     qs_k, qs_k_rowwise, survival, trapezoid_crps_rows)

DATA = Path("/mnt/d/data/awfdata")
PAIR = DATA / "derived" / "paired" / "paired_2021.parquet"
REMAF = Path("/home/vhyang/zcode/awf/outputs/e5_baseline/rema/"
             "rema_features_by_station.csv")
IN = Path("/home/vhyang/zcode/awf/outputs/review_v2")
OUT = Path("/home/vhyang/zcode/awf/outputs/review_v3")
THR = 15.0
TAUS = np.round(np.concatenate([np.arange(0.05, 0.96, 0.05),
                                [0.97, 0.98, 0.99]]), 4)
SEED = 42
QNAMES = [f"lgbm_q{int(round(t*100)):02d}" for t in TAUS]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = "AWF_E8_SCORES_R1"

    # ---------- A. 冻结输入与键校验 ----------
    pred = pd.read_parquet(IN / "predictions_full.parquet")
    raw = np.load(IN / "raw_Q22.npy")
    Qmono = pred[QNAMES].values.astype(float)     # 交付(重排后)
    assert raw.shape == Qmono.shape
    raw_sorted = monotonize_rows(raw)
    assert np.allclose(raw_sorted, Qmono, atol=1e-12), \
        "raw_Q22 排序后与交付分位不一致"
    assert not pred.duplicated(["station", "init", "valid_time",
                                "lead_h"]).any()
    y = pred.obs.values
    n = len(y)
    dates = pd.to_datetime(pred.init).dt.date.values
    qcols = {}
    raw_keyed = pred[["station", "init", "valid_time", "lead_h",
                      "obs", "gfs"]].copy()
    for j, nm in enumerate(QNAMES):
        raw_keyed[nm.replace("lgbm_q", "raw_q")] = raw[:, j]
    raw_keyed.to_parquet(OUT / "raw_Q22_keyed.parquet", index=False)

    # 原始/重排双版本并入主预测交付(文件在末尾统一写出)
    pred["q50_raw"] = raw[:, list(TAUS).index(0.5)]

    # ---------- B. 统一完整分布 + 精确 CRPS ----------
    tN, qN = build_complete_Q(Qmono, TAUS)
    row_exact = exact_crps_rows(y, tN, qN)
    row_trap = trapezoid_crps_rows(y, tN, qN)
    crps_exact = float(row_exact.mean())
    # 旧端点规则精确积分: 回归核对评审独立值 1.2464638
    dq = np.median(np.diff(Qmono, axis=1), axis=1)
    tO = np.empty((n, len(TAUS) + 2))
    tO[:, 0], tO[:, 1:-1], tO[:, -1] = 0.0, TAUS[None, :], 1.0
    qO = np.empty_like(tO)
    qO[:, 0] = np.maximum(Qmono[:, 0] - dq, 0.0)
    qO[:, 1:-1] = Qmono
    qO[:, -1] = Qmono[:, -1] + dq
    crps_oldrule_exact = float(exact_crps_rows(y, tO, qO).mean())
    regcheck = dict(target=1.2464637836911592,
                    got=crps_oldrule_exact,
                    abs_diff=abs(crps_oldrule_exact - 1.2464637836911592))
    assert regcheck["abs_diff"] < 1e-5, f"回归核对失败 {regcheck}"
    # 非负诊断(统一规则已截0; 记录原始负值行)
    neg_rows = int((raw.min(axis=1) < 0).sum())
    neg_min = float(raw.min())
    # 完整 Q(含端点)单调校验
    assert (np.diff(qN, axis=1) >= -1e-12).all(), "统一分布不单调"

    # EMOS / EMOS-geo
    emos_mu, emos_sd = pred.emos_mu.values, pred.emos_sd.values
    row_emos = crps_normal_closed(y, emos_mu, emos_sd)

    # EMOS-geo 重拟合: 坡度填充固定训练期中位数 (V3-06)
    df = pd.read_parquet(PAIR).rename(columns={"t2m_x": "obs_t2m",
                                               "t2m_y": "t2m"})
    reg = pd.read_csv(DATA / "derived/station_qc/station_sensor_registry.csv")
    df = df.merge(reg[["station", "lat", "lon", "elevation", "terrain"]],
                  on="station", how="left")
    df["fcst_spd"] = np.hypot(df.u10, df.v10)
    df["month"] = df.valid_time.dt.month
    df = df.dropna(subset=["wspd"]).reset_index(drop=True)
    df = df.merge(pd.read_csv(REMAF), on="station", how="left")
    trn = df[df.month <= 8]
    fill_train = float(np.nanmedian(trn.slope5k.values))
    aff_trn = int(trn.slope5k.isna().sum())
    aff_te = int(df[df.month >= 10].slope5k.isna().sum())
    assert len(df[df.month >= 10]) == n
    pred = pred.merge(
        df[df.month >= 10][["station", "init", "valid_time", "lead_h",
                            "elevation", "slope5k"]],
        on=["station", "init", "valid_time", "lead_h"],
        how="left", validate="one_to_one")
    assert pred.slope5k.notna().sum() + aff_te == n
    gfs_v = pred.gfs.values
    eg_mu = np.full(n, np.nan)
    eg_sd = np.full(n, np.nan)
    eg_fit = []
    for lh in sorted(pred.lead_h.unique()):
        trl = trn[trn.lead_h == lh]
        Xe = trl.elevation.values / 1000.0
        Xs = np.nan_to_num(trl.slope5k.values, nan=fill_train) / 10.0

        def objg(p, trl=trl, Xe=Xe, Xs=Xs):
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
        m_ = (pred.lead_h == lh).values
        a, b, c1, c2, c3, dd = best
        sl_te = np.nan_to_num(pred.slope5k.values[m_],
                              nan=fill_train) / 10.0
        eg_mu[m_] = (a + b * gfs_v[m_] + c1 * pred.elevation.values[m_] / 1000.0
                     + c2 * sl_te)
        eg_sd[m_] = np.maximum(c3 + dd * gfs_v[m_], 1e-3)
        eg_fit.append(dict(lead_h=int(lh), params=[round(v, 6) for v in best],
                           crps_train=round(bv, 6), success=bool(r.success),
                           n_iter=int(r.nit)))
    row_eg = crps_normal_closed(y, eg_mu, eg_sd)
    pred["emosg_mu_v3"], pred["emosg_sd_v3"] = eg_mu, eg_sd
    (OUT / "emosg_fit_v3.json").write_text(json.dumps(dict(
        run_id=run_id, fill_rule="train-period median slope5k",
        fill_value=fill_train, affected_train_rows=aff_trn,
        affected_test_rows=aff_te, per_lead=eg_fit), ensure_ascii=False,
        indent=2))

    # ---------- C. 有限分位得分 / MAE 双版本 ----------
    idx3 = [list(TAUS).index(round(t, 4)) for t in (0.1, 0.5, 0.9)]
    Q3 = Qmono[:, idx3]
    T3 = np.array([0.1, 0.5, 0.9])
    emos_q = {t: emos_mu + norm.ppf(t) * emos_sd for t in T3}
    qs3_lgbm = qs_k(y, Q3, T3)
    qs3_emos = qs_k(y, np.column_stack([emos_q[t] for t in T3]), T3)
    qs22 = qs_k(y, Qmono, TAUS)
    mae_raw = float(np.abs(raw[:, list(TAUS).index(0.5)] - y).mean())
    mae_mono = float(np.abs(Qmono[:, list(TAUS).index(0.5)] - y).mean())
    mae_stnmos = float(np.abs(pred.stnmos.values - y).mean())

    # ---------- D. 事件概率(同一分布) + Brier + 可靠度 ----------
    p_lgbm = survival(THR, tN, qN)
    p_emos = 1 - norm.cdf((THR - emos_mu) / emos_sd)
    p_eg = 1 - norm.cdf((THR - eg_mu) / eg_sd)
    train_rate = float((trn.wspd > THR).mean())
    p_clim = np.full(n, train_rate)
    ev = (y > THR).astype(float)
    brier = {k: float(np.mean((p - ev) ** 2))
             for k, p in [("LGBM", p_lgbm), ("EMOS", p_emos),
                          ("EMOSgeo", p_eg), ("clim_train", p_clim)]}
    # 可靠度: 零概率箱纳入 + 守恒 + Wilson CI
    edges = [0.0, .02, .05, .1, .2, .4, .8, 1.0]
    rel_rows = []
    rel_ok = True
    for name, p in [("LGBM", p_lgbm), ("EMOS", p_emos), ("EMOSgeo", p_eg)]:
        bins = pd.cut(p, edges, include_lowest=True)
        for b_, g in pd.DataFrame(dict(b=bins, p=p, e=ev)).groupby(
                "b", observed=False):
            k = len(g)
            if k == 0:
                rel_rows.append(dict(method=name, bin=str(b_), n=0,
                                     mean_pred=np.nan, obs_freq=np.nan,
                                     ci_lo=np.nan, ci_hi=np.nan))
                continue
            ph = float(g.p.mean())
            x_, k_ = float(g.e.sum()), k
            z = 1.959963984540054
            den = 1 + z ** 2 / k_
            ctr = (x_ / k_ + z ** 2 / (2 * k_)) / den
            hw = z * np.sqrt(x_ / k_ * (1 - x_ / k_) / k_
                             + z ** 2 / (4 * k_ ** 2)) / den
            rel_rows.append(dict(method=name, bin=str(b_), n=k,
                                 mean_pred=round(ph, 5),
                                 obs_freq=round(x_ / k_, 5),
                                 ci_lo=round(max(ctr - hw, 0.0), 5),
                                 ci_hi=round(min(ctr + hw, 1.0), 5)))
        sub = pd.DataFrame(rel_rows)
        sub = sub[(sub.method == name)]
        ev_sum = float((sub.obs_freq.fillna(0) * sub.n).sum())
        if int(sub.n.sum()) != n or int(round(ev_sum)) != int(ev.sum()):
            rel_ok = False
    pd.DataFrame(rel_rows).to_csv(OUT / "reliability_v3.csv", index=False)
    assert rel_ok, "可靠度分箱计数/事件不守恒"

    # ---------- E. 截断 bootstrap ----------
    mono_q50 = Qmono[:, list(TAUS).index(0.5)]
    mae_stn = np.abs(pred.stnmos.values - y)
    okm = np.isfinite(pred.stnmos.values)
    diffs = {
        "MAE_mono_LGBM-stnMOS": (np.abs(mono_q50 - y) - mae_stn)[okm],
        "MAE_mono_LGBM-EMOS": np.abs(mono_q50 - y) - np.abs(emos_mu - y),
        "QS3_LGBM-EMOS": qs_k_rowwise(y, Q3, T3)
        - qs_k_rowwise(y, np.column_stack([emos_q[t] for t in T3]), T3),
        "CRPSexact_LGBM-EMOS": row_exact - row_emos,
        "CRPSexact_LGBM-EMOSgeo": row_exact - row_eg,
        "Brier_LGBM-EMOS": (p_lgbm - ev) ** 2 - (p_emos - ev) ** 2,
    }
    boot_rows = []
    stats_store = {}
    drawn_store = {}
    for name, dvec in diffs.items():
        dts = dates[okm] if name.endswith("stnMOS") else dates
        for bl in (1, 3, 7):
            st, dr = block_bootstrap(dvec, dts, bl, n_boot=1000, seed=SEED,
                                     return_drawn=True)
            lo, hi = np.percentile(st, [2.5, 97.5])
            nd_uniq = len(set(map(str, pd.unique(dts))))
            assert (dr == nd_uniq).all(), f"{name} b{bl} 抽中天数!=唯一天数"
            boot_rows.append(dict(metric=name, block_days=bl,
                                  mean_diff=round(float(dvec.mean()), 6),
                                  ci_lo=round(float(lo), 6),
                                  ci_hi=round(float(hi), 6),
                                  excl0=bool(lo > 0 or hi < 0),
                                  drawn_dates=int(dr[0])))
            stats_store[f"{name}_b{bl}"] = st
            drawn_store[f"{name}_b{bl}"] = dr
    pd.DataFrame(boot_rows).to_csv(OUT / "bootstrap_v3.csv", index=False)
    np.savez_compressed(OUT / "bootstrap_stats_v3.npz", **stats_store,
                        **{f"drawn_{k}": v for k, v in drawn_store.items()})

    # ---------- F. 列联表/阈值曲线/交叉/样本流(同一预测重导出) ----------
    event = y > THR
    warn_v = {"GFS_thr": pred.gfs.values > THR,
              "LGBM_q90": Qmono[:, list(TAUS).index(0.9)] > THR,
              "EMOS_q90": emos_q[0.9] > THR}

    def cont(warn, ev_):
        H = int((warn & ev_).sum())
        M = int((~warn & ev_).sum())
        F = int((warn & ~ev_).sum())
        C = int((~warn & ~ev_).sum())
        return H, M, F, C

    rows = []
    lead_vals = pred.lead_h.values
    for name, w in warn_v.items():
        H, M, F, C = cont(w, event)
        rows.append(dict(method=name, lead_h="all", n=n, H=H, M=M, F=F, C=C,
                         POD=round(H / max(H + M, 1), 3),
                         POFD=round(F / max(F + C, 1), 3),
                         FAR=round(F / max(F + H, 1), 3),
                         CSI=round(H / max(H + M + F, 1), 3)))
        for lh in sorted(pred.lead_h.unique()):
            m = lead_vals == lh
            Hl, Ml, Fl, Cl = cont(w[m], event[m])
            rows.append(dict(method=name, lead_h=int(lh), n=int(m.sum()),
                             H=Hl, M=Ml, F=Fl, C=Cl,
                             POD=round(Hl / max(Hl + Ml, 1), 3),
                             POFD=round(Fl / max(Fl + Cl, 1), 3),
                             FAR=round(Fl / max(Fl + Hl, 1), 3),
                             CSI=round(Hl / max(Hl + Ml + Fl, 1), 3)))
    cv = pd.DataFrame(rows)
    inv_ok = True
    for name in cv.method.unique():
        sub = cv[(cv.method == name) & (cv.lead_h != "all")]
        allr = cv[(cv.method == name) & (cv.lead_h == "all")].iloc[0]
        for k in ("H", "M", "F", "C"):
            if int(sub[k].sum()) != int(allr[k]):
                inv_ok = False
        if not (cv[cv.method == name].eval("H+M+F+C == n")).all():
            inv_ok = False
    cv.to_csv(OUT / "contingency_v3.csv", index=False)

    # 阈值曲线(POD-POFD)
    tc = []
    for name, score in (("GFS", pred.gfs.values),
                        ("LGBM_q90", Qmono[:, list(TAUS).index(0.9)]),
                        ("EMOS_q90", emos_q[0.9])):
        for thr_q in np.unique(np.round(np.quantile(score, np.linspace(
                0.01, 0.995, 99)), 2)):
            w = score > thr_q
            H, M, F, C = cont(w, event)
            if H + M == 0 or F + C == 0:
                continue
            tc.append(dict(method=name, threshold=float(thr_q), POD=H / (H + M),
                           POFD=F / (F + C), FAR=F / (F + H),
                           CSI=H / (H + M + F)))
    pd.DataFrame(tc).to_csv(OUT / "threshold_curves_v3.csv", index=False)

    # 交叉(原始精度 vs 重排后)
    cross_raw = (np.diff(raw, axis=1) < -1e-12).any(axis=1)
    viol = np.maximum(-np.diff(raw, axis=1), 0)
    (OUT / "crossing_v3.json").write_text(json.dumps(dict(
        raw_rows=int(cross_raw.sum()), raw_rate=round(float(cross_raw.mean()), 4),
        raw_max_violation=round(float(viol.max()), 4),
        raw_p99_violation=round(float(np.percentile(viol, 99)), 4),
        post_rows=0), ensure_ascii=False, indent=2))

    # 样本流(重导出, 数值应与 v2 一致)
    flow = dict(
        raw_paired_rows=74334, wspd_missing=16634, valid_rows=57700,
        unique_station_valid=29216, stations_year=50,
        test_rows=n, test_stations=int(pred.station.nunique()),
        train_rows=len(trn), val_rows=int((df.month == 9).sum()),
        train_event_rate=round(train_rate, 5),
        test_event_rate=round(float(ev.mean()), 5))
    (OUT / "sample_flow_v3.json").write_text(json.dumps(
        flow, ensure_ascii=False, indent=2))

    # ---------- G. 汇总 ----------
    pred.to_parquet(OUT / "predictions_v3.parquet", index=False)
    scores = dict(
        QS3_LGBM=round(qs3_lgbm, 4), QS3_EMOS=round(qs3_emos, 4),
        QS22_LGBM=round(qs22, 4),
        CRPS_exact_LGBM=round(crps_exact, 6),
        CRPS_exact_trapezoid_diag=round(float(row_trap.mean()), 6),
        CRPS_exact_maxrow_diff_trap=round(
            float(np.abs(row_exact - row_trap).max()), 6),
        CRPS_oldrule_exact_regression=regcheck,
        CRPS_EMOS=round(float(row_emos.mean()), 4),
        CRPS_EMOSgeo_refit=round(float(row_eg.mean()), 4),
        MAE_q50_raw=round(mae_raw, 4), MAE_q50_mono=round(mae_mono, 4),
        MAE_stnMOS=round(mae_stnmos, 4), MAE_stnMOS_n=int(okm.sum()),
        brier=brier,
        negative_raw_quantile_rows=neg_rows, negative_raw_min=round(neg_min, 6),
        cdf_rule=("统一: 节点=单调重排拟合分位(max(q,0)); Q(0)=0 线性至 "
                  "(q05,0.05); Q(1)=q99+0.01*(q99-q95)/0.04(末段斜率延续, "
                  "退化时=q99); CRPS 精确分段积分; 超越概率同一 Q 广义逆"))
    (OUT / "scores_summary_v3.json").write_text(json.dumps(dict(
        scores=scores, taus=list(map(float, TAUS)), thr=THR,
        contingency_invariants=bool(inv_ok),
        protocol=("V3-01/02/03/06: 唯一完整分布(e8_core.build_complete_Q)"
                  "+精确积分; 可靠度含零概率箱并守恒; bootstrap 截断到"
                  "唯一日期数; EMOS-geo 填充=训练期中位数")),
        ensure_ascii=False, indent=2))
    (OUT / "run_meta.json").write_text(json.dumps(dict(
        run_id=run_id, utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        core_sha256=hashlib.sha256(
            (Path(__file__).parent / "e8_core.py").read_bytes()).hexdigest(),
        seed=SEED, env="conda dm",
        inputs=dict(predictions="review_v2/predictions_full.parquet",
                    raw_Q22="review_v2/raw_Q22.npy")), ensure_ascii=False,
        indent=2))
    print(json.dumps(scores, ensure_ascii=False, indent=1))
    print(pd.DataFrame(boot_rows).to_string(index=False))
    print("[contingency invariants]", "PASS" if inv_ok else "FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
