#!/usr/bin/env python
"""AWF E9 — V4-01/02: 与主表同版本的派生统计 + POFD-cap 阈值操作点

运行号: AWF_E9_DERIVED_R1 (服务器端自编)

对应 V4-01/V4-02:
  - 覆盖率/lead/分月: 全部由 review_v3/predictions_v3.parquet 的
    22 分位重排预测计算(与表 2 同一版本), 不再混用旧三分位结果。
  - 阈值操作点: threshold_curves_v3.csv 上按 "POFD cap 内最高 POD"
    读取; 报告实现 POFD(各方法不等), 明确 cap 语义而非 matched。
  - 增益定义统一: gain = (MAE_model/MAE_ref - 1)*100, 负=误差下降。

输出: outputs/review_v3/derived_stats_v4.json
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

V3 = Path("/home/vhyang/zcode/awf/outputs/review_v3")
OUT = V3 / "derived_stats_v4.json"


def main() -> int:
    p = pd.read_parquet(V3 / "predictions_v3.parquet")
    y = p.obs.values
    q10, q50, q90 = (p.lgbm_q10.values, p.lgbm_q50.values,
                     p.lgbm_q90.values)
    gfs = p.gfs.values
    cov = dict(q10=float((y <= q10).mean()),
               q50=float((y <= q50).mean()),
               q90=float((y <= q90).mean()))
    mae_m = np.abs(q50 - y)
    mae_g = np.abs(gfs - y)
    lead = p.lead_h.values
    months = pd.to_datetime(p.valid_time).dt.month.values
    rel = lambda m, g: float((m / g - 1.0) * 100.0)
    by_lead = {}
    for tag, m in (("lead0", lead == 0), ("lead6_72", lead != 0)):
        by_lead[tag] = round(rel(mae_m[m].mean(), mae_g[m].mean()), 4)
    by_month = {}
    for mo, name in ((10, "Oct"), (11, "Nov"), (12, "Dec")):
        m = months == mo
        by_month[name] = round(rel(mae_m[m].mean(), mae_g[m].mean()), 4)

    tc = pd.read_csv(V3 / "threshold_curves_v3.csv")
    caps = {}
    for cap in (0.02, 0.05, 0.10):
        row = {}
        for meth, g in tc.groupby("method"):
            gg = g[g.POFD <= cap]
            if len(gg) == 0:
                continue
            r = gg.loc[gg.POD.idxmax()]
            row[meth] = dict(threshold=float(r.threshold),
                             POD=round(float(r.POD), 4),
                             realized_POFD=round(float(r.POFD), 5),
                             FAR=round(float(r.FAR), 4),
                             CSI=round(float(r.CSI), 4))
        caps[str(cap)] = row

    # 统一分布下零概率行数(V4-01 复核: 应为 11,327 行/8 事件)
    import sys
    sys.path.insert(0, "/home/vhyang/zcode/awf/scripts")
    from e8_core import build_complete_Q, survival
    TAUS = np.round(np.concatenate([np.arange(0.05, 0.96, 0.05),
                                    [0.97, 0.98, 0.99]]), 4)
    Qm = p[[f"lgbm_q{int(round(t*100)):02d}" for t in TAUS]].values
    tN, qN = build_complete_Q(Qm, TAUS)
    p_exc = survival(15.0, tN, qN)
    ev = (y > 15.0)
    zero_n = int((p_exc == 0.0).sum())
    zero_ev = int(ev[p_exc == 0.0].sum())
    doc = dict(
        run_id="AWF_E9_DERIVED_R1",
        utc=datetime.now(timezone.utc).isoformat(),
        source="review_v3/predictions_v3.parquet (22-quantile rearranged)",
        gain_definition="(MAE_model/MAE_ref - 1)*100, negative = error "
                        "reduction vs raw GFS",
        coverage=cov,
        mae_gain_by_lead=by_lead,
        mae_gain_by_month=by_month,
        pofd_caps=caps,
        cap_semantics=("highest POD among evaluated thresholds with POFD "
                       "<= cap; realized POFD differs across methods and "
                       "is reported per operating point; test-set sweep, "
                       "not a locked operational threshold"),
        exceedance_zero_rows=dict(
            n=zero_n, events=zero_ev,
            note="unified-CDF P(obs>15)==0 rows (the superseded CDF rule "
                 "gave 9,330 rows / 3 events; both are included in the "
                 "reliability table, which bins all 18,235 rows)"),
    )
    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=2))
    print(json.dumps(doc, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
