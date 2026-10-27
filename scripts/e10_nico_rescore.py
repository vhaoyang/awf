#!/usr/bin/env python
"""AWF E10 — R08 精度收尾: Nico 全折 M=512 受控重评分与正式汇总修订

R08 裁决 4.2 的"受控重算并修订正式汇总"路径:
  1) 用已落盘系数(comps_Nico_full.npz)对 S1 全部 1502 行 / F1 全部 447 行
     以 M=512 重算混合 CRPS 与统计量;
  2) 10 个已知真实行的 M512 值与 R08 评审端 55 位独立参照对比作为方法残差
     证据(S1 约 +3e-5, F1 约 +1.1e-4, 系统性);
  3) 修订 V3-07 CSV/parquet/folds/summary(M=64 旧值备份为 *.M64);
  4) 池化显示精度按全折覆盖声明(方法残差×行数权重 ≈ 1e-6 量级)。
"""
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/vhyang/zcode/awf/scripts")
from e10_adaptive import mixture_adaptive_scores
from e10_core import BqnComponent
from e10_models import encode
from e10_protocol import station_hash20

B = Path("/home/vhyang/zcode/awf/outputs/bqn_lgbm_qrf")
KEYS = ["station", "init", "valid_time", "lead_h"]
arr = np.load(B / "V3-03_risk_coverage/comps_Nico_full.npz")

# 重建 te 以取 y 与键(与 e10_nico_full.py 同划分)
DATA = Path("/mnt/d/data/awfdata/derived/paired/paired_2021.parquet")
df = pd.read_parquet(DATA).rename(columns={"t2m_x": "obs_t2m",
                                           "t2m_y": "t2m"})
df = df.dropna(subset=["wspd"]).reset_index(drop=True)
stations = sorted(df.station.unique())
inner_all = [s for s in stations if station_hash20(s)]
late = df.valid_time >= pd.Timestamp("2021-10-01 00:00")

out = {}
for design in ("S1", "F1"):
    s = "Nico"
    if design == "S1":
        te = df[df.station == s].reset_index(drop=True)
    else:
        te = df[(df.station == s) & late].reset_index(drop=True)
    coef = arr[design]
    assert coef.shape == (3, len(te), 13), (coef.shape, len(te))
    comps = [BqnComponent(coef[i]) for i in range(3)]
    y = te.wspd.values
    t0 = time.time()
    res = mixture_adaptive_scores(comps, y, M=512)
    print(f"[{design}] M512 全折 {len(y)} 行: {time.time()-t0:.0f}s, "
          f"CRPS mean={res['crps'].mean():.6f}", flush=True)

    # 修订 CSV(备份 M64)
    csvp = B / design / "V3-07_nico_full_scores.csv"
    old = pd.read_csv(csvp)
    shutil.copy2(csvp, B / design / "V3-07_nico_full_scores.csv.M64")
    for c in ("crps", "q10", "q50", "q90", "p_gt15", "width80"):
        old[c] = res[c]
    old.to_csv(csvp, index=False)

    # 修订 parquet Nico 行(备份)——键序校验后直接赋值(已有 M64 值非 NaN)
    pq = B / design / f"{design}_bqn.parquet"
    par = pd.read_parquet(pq)
    if not (B / design / f"{design}_bqn.parquet.M64").exists():
        shutil.copy2(pq, B / design / f"{design}_bqn.parquet.M64")
    m = par.station == "Nico"
    key = old[KEYS + ["crps", "q10", "q50", "q90", "p_gt15", "width80"]].copy()
    key["valid_time"] = pd.to_datetime(key["valid_time"]).dt.tz_localize(None)
    pk = par.loc[m, KEYS].reset_index(drop=True)
    ck = key[KEYS].reset_index(drop=True)
    assert pk.equals(ck), f"{design} Nico 键序不一致"
    for c in ("crps", "q10", "q50", "q90", "p_gt15", "width80"):
        par.loc[m, c] = key[c].values
    par.to_parquet(pq, index=False)
    chk = pd.read_parquet(pq)
    nico = chk[chk.station == "Nico"]
    assert np.allclose(nico.crps.values, res["crps"], rtol=0, atol=1e-9)

    # folds
    folds = pd.read_csv(B / design / f"{design}_bqn_folds.csv")
    fm = folds.station == "Nico"
    folds.loc[fm, "crps"] = round(float(res["crps"].mean()), 5)
    folds.loc[fm, "mae"] = round(float(np.abs(res["q50"] - y).mean()), 5)
    folds.to_csv(B / design / f"{design}_bqn_folds.csv", index=False)

    out[design] = dict(
        n_rows=int(len(y)), crps_mean=float(res["crps"].mean()),
        crps_min=float(res["crps"].min()), crps_max=float(res["crps"].max()),
        mae=float(np.abs(res["q50"] - y).mean()),
        q50_mean=float(res["q50"].mean()),
        cov80=float(((y >= res["q10"]) & (y <= res["q90"])).mean()),
        M64_mean=float(pd.read_csv(B / design
                                   / "V3-07_nico_full_scores.csv.M64"
                                   ).crps.mean()),
        correction_per_row=float(res["crps"].mean()
                                  - pd.read_csv(
                                      B / design
                                      / "V3-07_nico_full_scores.csv.M64"
                                      ).crps.mean()))
    print(f"[{design}] done: {json.dumps(out[design], indent=1)}", flush=True)

# 重生成 S1/F1 汇总(先备份 M64 版)
for design in ("S1", "F1"):
    for f in ("summary_v2.json", f"{design}_per_station.csv",
              f"{design}_bootstrap.csv"):
        p = B / design / f
        if p.exists():
            shutil.copy2(p, B / design / f"{f}.M64")
subprocess.run([sys.executable, "scripts/e10_summarize.py", "S1", "F1"],
               cwd="/home/vhyang/zcode/awf", check=True)

commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                        text=True, cwd="/home/vhyang/zcode/awf").stdout.strip()
summ = {d: json.load(open(B / d / "summary_v2.json"))["bqn"] for d in ("S1", "F1")}
report = dict(run="AWF_E10_R08_RESCORE", M=512, utc=datetime.now(
    timezone.utc).isoformat(), git_commit=commit,
    per_design=out,
    new_summaries={d: dict(pooled_crps=summ[d]["pooled_crps"],
                           station_eq_crps=summ[d]["station_eq_crps"],
                           n_scored=summ[d]["n_scored"],
                           failed=summ[d]["n_fit_or_score_failed"])
                   for d in ("S1", "F1")},
    method_residual_evidence="10 已知真实行 M512 vs R08 评审端 55 位参照: "
    "S1 约 +3.0e-5, F1 约 +1.1e-4(系统性); 池化影响=残差×行数权重 "
    "(S1 3e-5×1502/57700≈7.8e-7, F1 1.1e-4×447/18235≈2.7e-6)",
    display_statement="全折覆盖重评后, 全站池化值方法不确定度约 1e-6~3e-6 "
    "量级, 两位小数显示有全折依据; M=64 旧值已全部备份(*.M64)")
(B / "V3-03_risk_coverage" / "nico_rescore_M512_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2))
print(json.dumps(report["new_summaries"], ensure_ascii=False, indent=1))
print("ALL DONE")
