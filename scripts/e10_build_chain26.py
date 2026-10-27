#!/usr/bin/env python
"""AWF E10 — V3-04 R2: 收口版本链(26号)构建

R05 缺口修复: 18 项预测(补 T-bqn + 文件/列映射), 24 配对得分与
Nico 早停记录内联, S1/F1 全站汇总与子集敏感性, V3-03 审计引用,
"South Pole"→"近南极点(lat −89°)" 措辞修正。
运行前须已 commit(引用固定 SHA), 产物只写讨论区。
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/vhyang/zcode/awf")
B = REPO / "outputs/bqn_lgbm_qrf"
DISC = Path("/mnt/c/Ian/NutDisk/RD/discuss/awf/05_投稿与归档/评审与修改"
            "/2026-10-24_WAF_manuscript_ams")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def jload(p):
    return json.loads(Path(p).read_text())


commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                        text=True, cwd=REPO).stdout.strip()
dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                       text=True, cwd=REPO).stdout.strip()

# ---- 源码 ----
source = {}
for f in ["e10_core.py", "e10_models.py", "e10_adaptive.py", "e10_run_T.py",
          "e10_run_S.py", "e10_run_F.py", "e10_run_F1.py", "e10_summarize.py",
          "e10_nico_full.py", "e10_audit_v303.py"]:
    p = REPO / "scripts" / f
    if p.exists():
        source[f] = dict(sha256=sha(p))

# ---- 搜索 + 24 配对(内联得分) ----
pairs = pd.read_csv(B / "T/V3-01_all_candidates.csv")
v301s = jload(B / "T/V3-01_summary.json")
search = dict(
    file=str((B / "T/search_results.json").relative_to(REPO)),
    sha256=sha(B / "T/search_results.json"),
    selected=jload(B / "T/search_results.json")["selected"],
    pairs_24_file=str((B / "T/V3-01_all_candidates.csv").relative_to(REPO)),
    pairs_24_sha256=sha(B / "T/V3-01_all_candidates.csv"),
    pairs_24_columns=pairs.columns.tolist(),
    pairs_24_rows=len(pairs),
    pairs_24_summary=v301s,
    note="24 配对=8 候选×3 种子的旧(cand4 时代评分)新(19-01/02 评分)验证 "
         "CRPS; 排名改变 cand4→cand0; T/T1 已按 cand0 重训(见 V3_01)")

# ---- 18 项预测 ----
P = {}


def pred(name, path, cols, gen, note):
    p = B / path
    P[name] = dict(file=str(p.relative_to(REPO)), sha256=sha(p),
                   columns=cols, generated_by=gen, note=note)


sc = ["crps", "q10", "q50", "q90", "p_gt15", "width80", "_grid_check"]
pred("T-bqn", "T/T_partial_BQN.parquet",
     ["station", "init", "valid_time", "lead_h", "obs", "gfs"] + sc[:6],
     "e10_run_T.py(官方混合步, cand0, commit 见 git 历史)",
     "BQN 三种子混合逐行评分; T_mixture.parquet 的 BQN_* 六列由本文件"
     "逐行复制(同键同序), 官方 T-BQN 数值以本文件为准")
pred("T-qrf", "T/T_partial_QRF.parquet", sc, "e10_run_T.py",
     "T_mixture.parquet QRF_* 六列同源")
pred("T-lgbm", "T/T_partial_LGBM.parquet", sc, "e10_run_T.py",
     "T_mixture.parquet LGBM_* 六列同源")
for d, g in (("S", "e10_run_S.py"), ("F", "e10_run_F.py")):
    pred(f"{d}-bqn", f"{d}/{d}_bqn.parquet",
         ["bqn_" + c for c in sc], g, "F0 13 列; 50/42 折全 ok")
    pred(f"{d}-qrf", f"{d}/{d}_qrf.parquet", ["bqn_" + c for c in sc], g, "")
    pred(f"{d}-lgbm", f"{d}/{d}_lgbm.parquet", ["bqn_" + c for c in sc], g, "")
for d in ("T1", "S1", "F1"):
    g = "e10_run_F1.py"
    n = {"T1": "地形消融 T(cand0)", "S1": "地形消融 S(预设 64/32)",
         "F1": "地形消融 F(预设 64/32)"}[d]
    extra = ""
    if d in ("S1", "F1"):
        extra = ("; Nico 折由 e10_nico_full.py 重训+无网格自适应评分后"
                 "全量回填(V3-07), 状态 ok-adaptive, 全网 0 NaN")
    pred(f"{d}-bqn", f"{d}/{d}_bqn.parquet", sc, g, n + extra)
    pred(f"{d}-qrf", f"{d}/{d}_qrf.parquet", sc, g, n)
    pred(f"{d}-lgbm", f"{d}/{d}_lgbm.parquet", sc, g, n)
assert len(P) == 18, len(P)

# ---- 汇总 ----
S = {}
for name, path in (("T_scores", "T/T_scores.json"),
                   ("T_bootstrap", "T/T_bootstrap.csv"),
                   ("S_summary", "S/summary_v2.json"),
                   ("F_summary", "F/summary_v2.json"),
                   ("S1_summary", "S1/summary_v2.json"),
                   ("F1_summary", "F1/summary_v2.json"),
                   ("S1_summary_subset_backup", "S1/summary_v2.json.subset_backup"),
                   ("F1_summary_subset_backup", "F1/summary_v2.json.subset_backup")):
    p = B / path
    if p.exists():
        S[name] = dict(file=str(p.relative_to(REPO)), sha256=sha(p))
s1s, f1s = jload(B / "S1/summary_v2.json"), jload(B / "F1/summary_v2.json")
full_network = dict(
    S1_bqn=dict(pooled_crps=s1s["bqn"]["pooled_crps"],
                station_eq_crps=s1s["bqn"]["station_eq_crps"],
                n_scored=s1s["bqn"]["n_scored"],
                failed=s1s["bqn"]["n_fit_or_score_failed"],
                subset_backup="S1/summary_v2.json.subset_backup(旧 49 站成功子集)"),
    F1_bqn=dict(pooled_crps=f1s["bqn"]["pooled_crps"],
                station_eq_crps=f1s["bqn"]["station_eq_crps"],
                n_scored=f1s["bqn"]["n_scored"],
                failed=f1s["bqn"]["n_fit_or_score_failed"],
                subset_backup="F1/summary_v2.json.subset_backup(旧 41 站成功子集)"),
    note="R04 裁决 3 口径: 全站为主, 旧共同成功子集为敏感性对照")

# ---- V3-01 / V3-03 / V3-07 ----
V301 = dict(
    evidence_file=str((B / "V3-01_validation_path_evidence.json")
                      .relative_to(REPO)),
    evidence_sha256=sha(B / "V3-01_validation_path_evidence.json"),
    summary="S/F/S1/F1 的 BQN parquet+folds 均生成于 19-01/02 修改之后"
            "(10-26 15:54/17:36/19:10/20:53, 早于 fdbf301 提交 21:06), "
            "训练期验证与早停基于修正后 _val_crps; Nico 折 R05 重训加无网格"
            "回退后早停工作(种子评估次数与 best_val_crps 见 evidence 文件)")
V303 = dict(
    audit_dir="outputs/bqn_lgbm_qrf/V3-03_risk_coverage",
    audit_summary_sha256=(sha(B / "V3-03_risk_coverage/summary.json")
                          if (B / "V3-03_risk_coverage/summary.json").exists()
                          else None),
    reference="e10_adaptive(分位像边界 panel-Simpson + head/tail 闭式修正 + "
              "Bernstein 解析矩 + τ域 GL), 自收敛=M 翻倍; 模块合成验证对"
              "恒等式参考 E|Xi-Xj|=E[Xi]+E[Xj]-2E[min] 最大相对误差 2.5e-6")
V307 = {}
for d in ("S1", "F1"):
    sj = jload(B / f"{d}/V3-07_nico_full_summary.json")
    V307[d] = dict(
        n_rows=sj["n_rows"], crps_mean=sj["crps_mean"], mae=sj["mae"],
        cov80=sj["cov80"], q50_mean=sj["q50_mean"],
        self_conv_max=sj["self_conv_max"],
        self_conv_rel_max=sj["self_conv_rel_max"], seeds=sj["seeds"],
        scores_csv=str((B / f"{d}/V3-07_nico_full_scores.csv")
                       .relative_to(REPO)),
        scores_csv_sha256=sha(B / f"{d}/V3-07_nico_full_scores.csv"),
        summary_json_sha256=sha(B / f"{d}/V3-07_nico_full_summary.json"))
V307["wording"] = ("Nico 站 lat=-89.0 为近南极点(非地理南极点 -90°); "
                   "旧 V3-07_nico_diagnosis.json 中 'South Pole' 措辞以本链为准修正")
V307["interpretation"] = ("三种子分布有限/非负/单调(合法); 所评行中位数偏低且"
                          "分布极宽(CRPS 很高), 为固定协议下的性能结果, "
                          "非'数学上无效'")

chain = dict(
    title="26_V3收口执行版本链",
    git_commit_full=commit, git_dirty=bool(dirty),
    generated=datetime.now(timezone.utc).isoformat(),
    note=f"固定引用 {commit[:12]}; 本次交付含 R05 四项缺口的补齐证据",
    source_code=source, search=search, predictions=P, summaries=S,
    full_network_summaries=full_network,
    V3_01=V301, V3_03=V303, V3_07=V307)

out = DISC / "26_V3收口执行版本链.json"
out.write_text(json.dumps(chain, ensure_ascii=False, indent=2))
print("written:", out)
print("commit:", commit, "dirty:", bool(dirty), "predictions:", len(P))
