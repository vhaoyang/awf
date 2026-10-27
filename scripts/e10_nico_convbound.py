#!/usr/bin/env python
"""AWF E10 — R09 §3.1: 全折积分余项覆盖说明(替代 10 行样本×权重)

对 S1/F1 全部 Nico 行计算 M=256 与 M=1024 两档, 与已交付 M=512 构成
三档全折收敛序列:
  - 逐行 |M512-M256| 与 |M1024-M512|; 方向一致时以 |M1024-M512|
    (不再外推) 作为 M512 残差的工作界, 与 R09 显示精度预算
    (S1 0.0999 / F1 0.1306)对比 => 两位小数舍入稳定性的全折覆盖证据;
  - 同时给出若改用 M1024 的池化位移(信息性, 不改交付)。
修正 nico_rescore_M512_report.json 的 M64_mean/correction 字段(从 *.M64
备份读取), 并区分"旧值->M512 变化"与"旧值->独立参照差"两个量。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/vhyang/zcode/awf/scripts")
from e10_adaptive import mixture_crps_only
from e10_core import BqnComponent

B = Path("/home/vhyang/zcode/awf/outputs/bqn_lgbm_qrf")
arr = np.load(B / "V3-03_risk_coverage/comps_Nico_full.npz")
DATA = Path("/mnt/d/data/awfdata/derived/paired/paired_2021.parquet")
df = pd.read_parquet(DATA).dropna(subset=["wspd"]).reset_index(drop=True)
late = df.valid_time >= pd.Timestamp("2021-10-01 00:00")

budget = {"S1": 0.0999, "F1": 0.1306}
other_sum = {"S1": (84458.5208, 57700), "F1": (27305.4050, 18235)}
conv, rep_fix = {}, {}
rep_p = B / "V3-03_risk_coverage/nico_rescore_M512_report.json"
rep = json.loads(rep_p.read_text())

for design in ("S1", "F1"):
    te = df[df.station == "Nico"] if design == "S1" else \
        df[(df.station == "Nico") & late]
    y = te.wspd.values
    coef = arr[design]
    comps = [BqnComponent(coef[i]) for i in range(3)]
    m512 = pd.read_csv(B / design / "V3-07_nico_full_scores.csv").crps.values
    t0 = time.time()
    m256 = mixture_crps_only(comps, y, M=256)
    print(f"[{design}] M256 {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    m1024 = mixture_crps_only(comps, y, M=1024)
    print(f"[{design}] M1024 {time.time()-t0:.0f}s", flush=True)
    d_lo = np.abs(m512 - m256)
    d_hi = np.abs(m1024 - m512)
    ratio = float(np.median(d_lo / np.maximum(d_hi, 1e-15)))
    n_sum, N = other_sum[design]
    pooled512 = (m512.sum() + n_sum) / N
    pooled1024 = (m1024.sum() + n_sum) / N
    conv[design] = dict(
        n_rows=int(len(y)),
        max_abs_M256_M512=float(d_lo.max()),
        max_abs_M512_M1024=float(d_hi.max()),
        median_ratio_lo_over_hi=ratio,
        direction_consistent=bool(np.all(np.sign(m512 - m256)
                                         == np.sign(m1024 - m512))),
        working_residual_bound=float(d_hi.max()),
        budget=budget[design],
        bound_within_budget=bool(d_hi.max() <= budget[design]),
        pooled_M512_delivered=round(float(pooled512), 6),
        pooled_if_m1024=round(float(pooled1024), 6),
        pooled_shift=round(float(pooled1024 - pooled512), 8))
    print(f"[{design}] {json.dumps(conv[design], indent=1)}", flush=True)
    # 报告字段修正
    m64 = pd.read_csv(B / design / "V3-07_nico_full_scores.csv.M64")
    rep["per_design"][design]["M64_mean"] = float(m64.crps.mean())
    rep["per_design"][design]["correction_per_row"] = float(
        m512.mean() - m64.crps.mean())
    rep["per_design"][design][
        "note"] = ("correction_per_row = M64 交付值 -> M512 交付值的每行均值"
                   "变化; 与 R08 表中'旧值到独立参照的差'是不同量")

rep["field_fix_note"] = ("R09 §3.2 更正: 原 M64_mean/correction_per_row 在"
                         "二次运行时误读已更新 CSV; 现从 *.M64 备份重读修正")
rep["full_fold_convergence"] = conv
rep["method_residual_evidence"] = (
    "全折收敛序列(M256/M512/M1024, 全部 Nico 行): M512 工作残差界 = "
    "max|M1024-M512|(不外推); 与 R09 预算对比见 full_fold_convergence")
rep_p.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
print("DONE")
