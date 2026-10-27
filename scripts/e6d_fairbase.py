#!/usr/bin/env python
"""AWF E6D — 评审 P1：公平基线 + 相关不确定性 + 预警阈值曲线 + Brier（R3.1/R1.4/R1.3）

运行号: AWF_E6D_FAIRBASE_R1 (服务器端自编, 非本侧任务号)

内容:
  (1) 信息可比基线(时间块口径): 站点×lead MOS; 站点气候分位数(逐站逐lead
      经验分位, 训练期拟合, 常数分位预报); EMOS 非负性核查 + 对数正态
      EMOS 变体(观测~LogNormal, CRPS 数值积分最小化)。
  (2) 块 bootstrap: 按起报日分块(同日跨站跨lead相关保留), 对
      MAE(LGBM − 对手) 与 CRPS3(LGBM − EMOS) 给 95% CI;
      块长敏感性 {1,3,7} 天(移动块)。
  (3) 阈值曲线: GFS/QM(点值阈值) 与 LGBM/EMOS(q90 阈值) 的
      POD/POFD/FAR-vs-阈值; 匹配 POFD∈{.02,.05,.10} 的 POD 对照。
  (4) Brier + 可靠度: P(y>15) —— EMOS 用闭式; LGBM 用三分位线性
      插值 CDF(近似, 声明)。
输出: outputs/review_p1/{fair_baselines.csv, bootstrap_ci.csv,
       threshold_curves.csv, matched_pofd.csv, brier_reliability.csv,
       p1_fair_summary.json}
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import lognorm, norm

DATA = Path("/mnt/d/data/awfdata")
PAIR = DATA / "derived" / "paired" / "paired_2021.parquet"
TPRED = Path("/home/vhyang/zcode/awf/outputs/e5_baseline/test_predictions.parquet")
OUT = Path("/home/vhyang/zcode/awf/outputs/review_p1")
THR = 15.0
SEED = 42
QS = [0.1, 0.5, 0.9]


def crps_normal(y, m, s):
    z = (y - m) / s
    return s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def fit_emos(x, y, logn=False):
    if logn:
        # Lognormal ML 回归: log(y) ~ a + b*log1p(x) (闭式 OLS)
        ly = np.log(np.maximum(y, 0.05))
        lx = np.log1p(np.maximum(x, 0))
        b, a = np.polyfit(lx, ly, 1)
        res = ly - (a + b * lx)
        return np.array([a, b, np.log(res.std() + 1e-3)])
    def obj(p):
        a, b, c, d = p
        if c <= 0:
            return 1e9
        s = np.maximum(c + d * x, 1e-3)
        return float(np.mean(crps_normal(y, a + b * x, s)))
    best, bv = None, np.inf
    sy = y.std()
    for m0 in ([y.mean(), 1.0, sy, 0.0], [y.mean(), 0.9, .5 * sy, .05],
               [0.0, 1.0, sy, 0.0]):
        r = minimize(obj, m0, method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-4, fatol=1e-5))
        if r.fun < bv:
            bv, best = r.fun, r.x
    return best


def _crps_lognorm(y, mu, sig):
    """Lognormal CRPS (数值稳健近似: 按 400 分位格点积分)。"""
    qs = np.linspace(0.001, 0.999, 400)
    xq = lognorm.ppf(qs, sig, scale=np.exp(mu))
    w = np.diff(np.concatenate([[0], qs, [1]]))
    out = np.zeros_like(y, dtype=float)
    # CRPS = 2*E|X-y| - E|X-X'|
    exy = (np.abs(xq[None, :] - y[:, None]) * w[1:-1][None, :]).sum(1)
    m = (w[1:-1] * xq).sum()
    var2 = 2 * (w[1:-1] * np.abs(xq - m)).sum()
    return 2 * exy - var2


def pinball(y, q, tau):
    z = y - q
    return np.maximum(tau * z, (tau - 1) * z)


def crps3(y, d):
    return float(2 * np.mean([np.mean(pinball(y, d[t], t)) for t in QS]))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = "AWF_E6D_FAIRBASE_R1"
    rng = np.random.default_rng(SEED)
    tp = pd.read_parquet(TPRED)
    df = pd.read_parquet(PAIR).rename(columns={"t2m_x": "obs_t2m",
                                               "t2m_y": "t2m"})
    df["fcst_spd"] = np.hypot(df.u10, df.v10)
    df["month"] = df.valid_time.dt.month
    df = df.dropna(subset=["wspd"]).reset_index(drop=True)
    trn = df[df.month <= 8].reset_index(drop=True)
    te = df[df.month >= 10].reset_index(drop=True)
    assert len(te) == len(tp)

    # ---------- (1) 公平基线 ----------
    y = te.wspd.values
    preds = {"LGBM": dict(zip(QS, [tp[f"lgbm_spd_q{int(t*100)}"].values
                                   for t in QS]))}
    # 站点×lead MOS
    smos = np.full(len(te), np.nan)
    for (s, lh), g_tr in trn.groupby(["station", "lead_h"]):
        if len(g_tr) < 20 or g_tr.fcst_spd.std() < 1e-6:
            continue
        b, a = np.polyfit(g_tr.fcst_spd, g_tr.wspd, 1)
        m = (te.station == s) & (te.lead_h == lh)
        smos[m] = a + b * te.fcst_spd[m]
    preds["stnMOS"] = {0.5: smos}
    # 站点气候分位数(逐站逐lead, <20 行回退逐站)
    clim = {t: np.full(len(te), np.nan) for t in QS}
    for s in te.station.unique():
        gs = trn[trn.station == s]
        for lh in te.lead_h.unique():
            g = gs[gs.lead_h == lh]
            if len(g) < 20:
                g = gs
            m = (te.station == s) & (te.lead_h == lh)
            for t in QS:
                clim[t][m] = np.quantile(g.wspd, t)
    preds["stnClimQ"] = clim
    # EMOS(重拟合, 与 E5I 同) + 对数正态 EMOS + 非负性核查
    emos_q = {t: np.full(len(te), np.nan) for t in QS}
    lg_q = {t: np.full(len(te), np.nan) for t in QS}
    n_neg_q10 = 0
    for lh in sorted(te.lead_h.unique()):
        trl = trn[trn.lead_h == lh]
        p = fit_emos(trl.fcst_spd.values, trl.wspd.values)
        m = te.lead_h == lh
        mu = p[0] + p[1] * te.fcst_spd[m]
        sd = np.maximum(p[2] + p[3] * te.fcst_spd[m], 1e-3)
        for t in QS:
            emos_q[t][m] = mu + norm.ppf(t) * sd
        n_neg_q10 += int((emos_q[0.1][m] < 0).sum())
        # 对数正态变体
        pl = fit_emos(trl.fcst_spd.values, trl.wspd.values, logn=True)
        mlu = pl[0] + pl[1] * np.log1p(te.fcst_spd[m])
        lsig = np.exp(pl[2])
        for t in QS:
            lg_q[t][m] = lognorm.ppf(t, lsig, scale=np.exp(mlu))
    preds["EMOS"] = emos_q
    preds["EMOSlogn"] = lg_q

    rows = []
    for name, d in preds.items():
        q50 = d[0.5]
        ok = ~np.isnan(q50)
        row = dict(method=name, n=int(ok.sum()),
                   mae=float(np.abs(q50[ok] - y[ok]).mean()),
                   bias=float((q50[ok] - y[ok]).mean()))
        if all(t in d for t in QS) and not np.isnan(d[0.1]).all():
            ok3 = ok & ~np.isnan(d[0.1]) & ~np.isnan(d[0.9])
            row["crps3"] = crps3(y[ok3], {t: d[t][ok3] for t in QS})
        rows.append(row)
    fair = pd.DataFrame(rows).round(4)
    fair.to_csv(OUT / "fair_baselines.csv", index=False)
    print(fair.to_string(index=False))

    # ---------- (2) 块 bootstrap ----------
    init_date = pd.to_datetime(te.init).dt.date.values
    lgbm50 = tp["lgbm_spd_q50"].values
    allidx = np.arange(len(y))
    m_ok = ~np.isnan(smos)          # stnMOS 仅覆盖子集, 用共同子集配对
    diffs = [
        ("MAE_LGBM-stnMOS",
         (np.abs(lgbm50 - y) - np.abs(smos - y))[m_ok], allidx[m_ok]),
        ("MAE_LGBM-EMOS",
         np.abs(lgbm50 - y) - np.abs(emos_q[0.5] - y), allidx),
        ("CRPS3_LGBM-EMOS",
         np.concatenate([pinball(y, tp.lgbm_spd_q10, .1)
                         - pinball(y, emos_q[.1], .1),
                         pinball(y, tp.lgbm_spd_q50, .5)
                         - pinball(y, emos_q[.5], .5),
                         pinball(y, tp.lgbm_spd_q90, .9)
                         - pinball(y, emos_q[.9], .9)]),
         np.concatenate([allidx] * 3)),
    ]
    boot_rows = []
    for name, d_arr, idx_arr in diffs:
        blocks_by_date = {}
        for i, dt in enumerate(init_date):
            blocks_by_date.setdefault(dt, []).append(i)
        for blen in (1, 3, 7):
            d_sorted = np.array(sorted(blocks_by_date.keys()))
            nb = len(d_sorted)
            stats = []
            for _ in range(1000):
                sel = []
                while len(sel) * blen < nb + blen:
                    st = rng.integers(0, nb - blen + 1)
                    sel.extend(d_sorted[st:st + blen])
                rows_i = np.concatenate([blocks_by_date[d] for d in sel])
                mask = np.isin(idx_arr, rows_i)
                stats.append(d_arr[mask].mean())
            stats = np.array(stats)
            pt = d_arr.mean()
            boot_rows.append(dict(metric=name, block_days=blen,
                                  mean_diff=round(float(pt), 4),
                                  ci95_lo=round(float(np.percentile(stats, 2.5)), 4),
                                  ci95_hi=round(float(np.percentile(stats, 97.5)), 4),
                                  sig_0_excluded=bool(
                                      np.percentile(stats, 2.5) > 0
                                      or np.percentile(stats, 97.5) < 0)))
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(OUT / "bootstrap_ci.csv", index=False)
    print(boot.to_string(index=False))

    # ---------- (3) 阈值曲线 + 匹配 POFD ----------
    event = y > THR
    warn_vars = {
        "GFS": tp.gfs_spd.values,
        "QM": tp.qm_spd.values,
        "LGBM_q90": tp.lgbm_spd_q90.values,
        "EMOS_q90": emos_q[0.9],
    }
    thr_grid = np.arange(6.0, 30.01, 0.5)
    tc_rows = []
    for name, v in warn_vars.items():
        for t in thr_grid:
            w = v > t
            H = int((w & event).sum()); M = int((~w & event).sum())
            F = int((w & ~event).sum()); C = int((~w & ~event).sum())
            tc_rows.append(dict(method=name, threshold=float(t),
                                POD=H / max(H + M, 1), POFD=F / max(F + C, 1),
                                FAR=F / max(F + H, 1)))
    tcur = pd.DataFrame(tc_rows)
    tcur.to_csv(OUT / "threshold_curves.csv", index=False)
    matched = []
    for tpofd in (0.02, 0.05, 0.10):
        for name in warn_vars:
            sub = tcur[(tcur.method == name)
                       & (tcur.POFD <= tpofd + 1e-9)]
            if len(sub):
                # POFD<=目标 内 POD 最大(最小可行阈值)
                r = sub.loc[sub.POD.idxmax()]
                matched.append(dict(target_POFD=tpofd, method=name,
                                    threshold=float(r.threshold),
                                    POD=round(float(r.POD), 3),
                                    POFD=round(float(r.POFD), 3),
                                    FAR=round(float(r.FAR), 3)))
    pd.DataFrame(matched).to_csv(OUT / "matched_pofd.csv", index=False)
    print(pd.DataFrame(matched).to_string(index=False))

    # (4) Brier/可靠度改由 AWF_E6E_DENSEQ_R1 密分位方案承担

    (OUT / "p1_fair_summary.json").write_text(json.dumps(dict(
        run_id=run_id, utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fair_baselines=fair.to_dict("records"),
        emos_negative_q10_count=n_neg_q10,
        note="stnMOS/stnClimQ 仅训练期拟合; 块=起报日(保留同日跨站跨lead相关); "
             "LGBM 概率=三分位两段线性插值近似(声明); matched_pofd 取"
             "POFD<=目标下最大阈值", ensure_ascii=False, indent=2)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
