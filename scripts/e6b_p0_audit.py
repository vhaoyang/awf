#!/usr/bin/env python
"""AWF E6B — 评审 P0 核查：强风列联表 + 配对/切分审计 + 0h 单列（R1.3/R3.2）

运行号: AWF_E6B_P0AUDIT_R1 (服务器端自编, 非本侧任务号)

依据模拟评审 R1.3/R3.2：
  (a) 强风(观测>15 m/s)列联表: 逐模型(GFS阈值平移/QM/LGBM/EMOS) 逐 lead 与全量,
      H/M/F/C + 基准率 + POD/POFD/FAR/CSI。EMOS 需重拟合存预测(确定性, 秒级)。
      注: GFS/QM 无概率输出, 阈值平移=点预报>15; LGBM/EMOS 用 q90>15
      (连续校准下约对应"超阈概率>10%"的低概率预警规则)。
  (b) 配对与切分审计: (station,init,lead) 唯一性; (station,valid_time) 重复;
      训练/验证/测试 valid_time 两两交集(标签泄漏面); 按有效时间 vs 起报时间
      划分的边界溢出(72h lead 跨界); lead 0(分析时效)单独指标。
  (c) 0h 与 6-72h 分列的 LGBM/GFS MAE/偏差。

输出: outputs/review_p0/{contingency.csv, pairing_audit.json,
       lead0_split.csv, p0_summary.json}
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

DATA = Path("/mnt/d/data/awfdata")
PAIR = DATA / "derived" / "paired" / "paired_2021.parquet"
TPRED = Path("/home/vhyang/zcode/awf/outputs/e5_baseline/test_predictions.parquet")
OUT = Path("/home/vhyang/zcode/awf/outputs/review_p0")
THR = 15.0


def crps_normal(y, m, s):
    z = (y - m) / s
    return s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def fit_emos(x, y):
    def obj(p):
        a, b, c, d = p
        if c <= 0:
            return 1e9
        s = np.maximum(c + d * x, 1e-3)
        return float(np.mean(crps_normal(y, a + b * x, s)))

    best, bv = None, np.inf
    sy = y.std()
    for m0 in ([y.mean(), 1.0, sy, 0.0], [y.mean(), 0.9, 0.5 * sy, 0.05],
               [0.0, 1.0, sy, 0.0]):
        r = minimize(obj, m0, method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-4, fatol=1e-5))
        if r.fun < bv:
            bv, best = r.fun, r.x
    return best


def cont_row(method, lead, warn, event):
    H = int(((warn) & (event)).sum())
    M = int(((~warn) & (event)).sum())
    F = int((warn & ~event).sum())
    C = int(((~warn) & (~event)).sum())
    return dict(method=method, lead_h=("all" if lead is None else int(lead)),
                n=H + M + F + C, events=H + M, H=H, M=M, F=F, C=C,
                base_rate=round((H + M) / max(H + M + F + C, 1), 4),
                POD=round(H / max(H + M, 1), 3),
                POFD=round(F / max(F + C, 1), 3),
                FAR=round(F / max(F + H, 1), 3),
                CSI=round(H / max(H + M + F, 1), 3))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = "AWF_E6B_P0AUDIT_R1"
    tp = pd.read_parquet(TPRED)
    # EMOS 重拟合(确定性)取 q90 —— 与 E5I 同协议
    df = pd.read_parquet(PAIR).rename(columns={"t2m_x": "obs_t2m",
                                               "t2m_y": "t2m"})
    df["fcst_spd"] = np.hypot(df.u10, df.v10)
    df["month"] = df.valid_time.dt.month
    df = df.dropna(subset=["wspd"]).reset_index(drop=True)
    trn = df[df.month <= 8]
    emos_q90 = np.full(len(tp), np.nan)
    for lh in sorted(tp.lead_h.unique()):
        trl = trn[trn.lead_h == lh]
        p = fit_emos(trl.fcst_spd.values, trl.wspd.values)
        m_ = tp[tp.lead_h == lh]
        mu = p[0] + p[1] * m_.gfs_spd.values
        sd = np.maximum(p[2] + p[3] * m_.gfs_spd.values, 1e-3)
        emos_q90[m_.index.values] = mu + norm.ppf(0.9) * sd
    tp["emos_q90"] = emos_q90

    event = tp.wspd.values > THR
    methods = [
        ("GFS_thr", tp.gfs_spd.values > THR),
        ("QM_thr", tp.qm_spd.values > THR),
        ("LGBM_q90", tp.lgbm_spd_q90.values > THR),
        ("EMOS_q90", tp.emos_q90.values > THR),
    ]
    rows = []
    for name, warn in methods:
        rows.append(cont_row(name, None, warn, event))
        for lh in sorted(tp.lead_h.unique()):
            m = tp.lead_h.values == lh
            rows.append(cont_row(name, lh, warn & m, event & m))
    cont = pd.DataFrame(rows)
    cont.to_csv(OUT / "contingency.csv", index=False)
    print(cont[cont.lead_h == "all"].to_string(index=False))

    # ---- (b) 配对/切分审计 ----
    audit = {}
    audit["n_rows"] = len(df)
    audit["unique_station_init_lead"] = int(
        ~df.duplicated(["station", "init", "lead_h"]).any())
    audit["dup_station_valid_lead"] = int(df.duplicated(
        ["station", "valid_time", "lead_h"]).sum())
    audit["n_unique_station_valid"] = int(df.groupby(
        ["station", "valid_time"]).ngroups)
    audit["n_independent_obs_station_hours"] = int(
        df.drop_duplicates(["station", "valid_time"]).shape[0])
    vm = df.valid_time
    tr_va = set(vm[(df.month <= 8)].dt.date) & set(vm[df.month == 9].dt.date)
    tr_te = set(vm[(df.month <= 8)].dt.date) & set(vm[df.month >= 10].dt.date)
    va_te = set(vm[df.month == 9].dt.date) & set(vm[df.month >= 10].dt.date)
    audit["valid_time_overlap_train_val"] = len(tr_va)
    audit["valid_time_overlap_train_test"] = len(tr_te)
    audit["valid_time_overlap_val_test"] = len(va_te)
    # 起报时间口径: init 月 vs 有效月 (72h 跨界)
    df["init_month"] = pd.to_datetime(df.init).dt.month
    cross = df[(df.init_month <= 8) & (df.month >= 9)]
    audit["rows_init_in_train_valid_after"] = int(len(cross))
    cross2 = df[(df.init_month == 9) & (df.month >= 10)]
    audit["rows_init_val_valid_in_test"] = int(len(cross2))
    # 测试集内部: 起报在9月(验证期)但有效在10-12月的行是否混入测试
    audit["test_rows_with_init_before_oct"] = int(
        ((df.month >= 10) & (df.init_month <= 9)).sum())

    # ---- (c) 0h 单列 ----
    l0 = []
    for scope, mask in [("lead0", tp.lead_h == 0),
                        ("lead6_72", tp.lead_h > 0)]:
        g = tp[mask]
        l0.append(dict(scope=scope, n=len(g),
                       gfs_mae=float(np.abs(g.gfs_spd - g.wspd).mean()),
                       lgbm_mae=float(np.abs(g.lgbm_spd_q50 - g.wspd).mean()),
                       gfs_bias=float((g.gfs_spd - g.wspd).mean()),
                       lgbm_bias=float((g.lgbm_spd_q50 - g.wspd).mean())))
    pd.DataFrame(l0).round(3).to_csv(OUT / "lead0_split.csv", index=False)
    print(pd.DataFrame(l0).round(3).to_string(index=False))

    (OUT / "pairing_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2))
    (OUT / "p0_summary.json").write_text(json.dumps(dict(
        run_id=run_id, utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        contingency_all=cont[cont.lead_h == "all"].to_dict("records"),
        pairing_audit=audit, lead0_split=l0,
        notes=dict(
            far_def="FAR=F/(F+H); POFD=F/(F+C); 稿件原文把 POFD 误标为 FAR",
            emos_repro="EMOS 重拟合(与 AWF_E5I_EMOS_R1 同协议)以取逐行 q90"),
    ), ensure_ascii=False, indent=2))
    print(json.dumps(audit, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
