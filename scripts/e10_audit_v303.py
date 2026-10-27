#!/usr/bin/env python
"""AWF E10 — V3-03 R05: 六设计×风险分层的三种子 BQN 混合精度审计

R04-03/R05 要求: 精度参照需覆盖六设计、各站、窄支撑/尾部/大支撑、
Nico 与生产批量入口, 给样本选择、输入身份、参照定义、误差摘要与
超限记录; 允许风险分层抽查, 不要求全行高代价参照。

方法(全部为风险分层抽样, 非全量):
  T/T1 : 官方配置整体重训(确定性种子) -> 测试集全行对比;
  S/S1/F/F1: 每设计抽 2 站(平均 width80 最窄 + 最宽, 排除 Nico),
         按折协议重训该站 -> 该站全部行对比;
  Nico(S1/F1): 已由 e10_nico_full.py 全量自适应评分, 本审计登记其
         自收敛上限并引用其产物;
  QRF/LGBM(S 设计 1 站): 确定性复现抽查(结构精确路径既有验收)。

每个 BQN 条目记录:
  reproduce: 重训组件经生产入口的 q50/crps vs 交付 parquet 的最大差
             (训练确定性证据);
  prod-vs-ref: 生产网格路径 vs e10_adaptive 参照(同一组件)的逐行差
             (生产精度证据);
  self_conv: 参照 M 翻倍自收敛(参照误差控制);
  exceed: |prod-ref| > 1e-4 的行数(超限记录)。

输出: outputs/bqn_lgbm_qrf/V3-03_risk_coverage/{audit_rows.csv,
       summary.json}
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/vhyang/zcode/awf/scripts")
from e10_adaptive import mixture_adaptive_scores
from e10_models import BQN, LGBM22, QRF, chunked_mixture_scores, encode
from e10_protocol import station_hash20

DATA = Path("/mnt/d/data/awfdata")
REMAF = Path("/home/vhyang/zcode/awf/outputs/e5_baseline/rema/"
             "rema_features_by_station.csv")
TFV3 = Path("/home/vhyang/zcode/awf/outputs/review_v3/terrain_features_v3.csv")
BASE = Path("/home/vhyang/zcode/awf/outputs/bqn_lgbm_qrf")
OUTD = BASE / "V3-03_risk_coverage"
OUTD.mkdir(exist_ok=True)
SEEDS = [17, 42, 2026]
KEYS = ["station", "init", "valid_time", "lead_h"]
F0 = ["u10", "v10", "fcst_spd", "t2m", "prmsl", "rh2m", "lead_h",
      "month_sin", "month_cos", "lat", "lon", "elevation", "terrain"]
F1_EXTRA = ["slope1k", "slope5k", "slope25k", "prominence", "fetch_drop",
            "katab_align_EN", "is_flat", "is_missing"]
F1 = F0 + F1_EXTRA
THR = 1e-4


def load(feat):
    df = pd.read_parquet(DATA / "derived/paired/paired_2021.parquet").rename(
        columns={"t2m_x": "obs_t2m", "t2m_y": "t2m"})
    reg = pd.read_csv(DATA / "derived/station_qc/station_sensor_registry.csv")
    df = df.merge(reg[["station", "lat", "lon", "elevation", "terrain"]],
                  on="station", how="left")
    df["terrain"] = df.terrain.astype("category")
    df["fcst_spd"] = np.hypot(df.u10, df.v10)
    df["month"] = df.valid_time.dt.month
    df["month_sin"] = np.sin(2 * np.pi * df.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.month / 12)
    df = df.dropna(subset=["wspd"]).reset_index(drop=True)
    if feat == "F1":
        df = df.merge(pd.read_csv(REMAF), on="station", how="left")
        tf = pd.read_csv(TFV3).set_index("station")
        dE = df.station.map(tf.downhill_E)
        dN = df.station.map(tf.downhill_N)
        w = np.maximum(np.hypot(df.u10, df.v10), 1e-9)
        df["katab_align_EN"] = (dE.values * df.u10.values
                                + dN.values * df.v10.values) / w
        st_map = df.station.map(tf.status)
        df["is_flat"] = (st_map == "flat").astype(int).values
        df["is_missing"] = (st_map == "missing").astype(int).values
        df.loc[df.is_flat == 1, "katab_align_EN"] = 0.0
        df.loc[df.is_missing == 1, "katab_align_EN"] = 0.0
    return df


def fill_train(df_tr, df_other_list):
    med = {}
    for c in F1_EXTRA[:5]:
        med[c] = float(df_tr[c].median())
    for d in [df_tr] + df_other_list:
        for c, v in med.items():
            d[c] = d[c].fillna(v)
    return med


def fold_split(df, design, s, inner_all, early, late):
    if design == "T":
        tr = df[df.month <= 8].reset_index(drop=True)
        iv = df[df.month == 9].reset_index(drop=True)
        te = df[df.month >= 10].reset_index(drop=True)
    elif design == "S":
        test = df.station == s
        inner = (~test) & df.station.isin([c for c in inner_all if c != s])
        tr = df[(~test) & (~inner)].reset_index(drop=True)
        iv = df[inner].reset_index(drop=True)
        te = df[test].reset_index(drop=True)
    else:
        test = (df.station == s) & late
        inner = ((df.station != s) & early & df.station.isin(
            [c for c in inner_all if c != s]) & (df.valid_time.dt.month == 8))
        tr = df[(df.station != s) & early & (~inner)].reset_index(drop=True)
        iv = df[inner].reset_index(drop=True)
        te = df[test].reset_index(drop=True)
    return tr, iv, te


def bqn_comps(tr, iv, te, feat, cfg, kw):
    comps = []
    for sd in SEEDS:
        m = BQN(seed=sd, **cfg, **kw).fit(
            encode(tr, feat, "bqn"), tr.wspd.values,
            encode(iv, feat, "bqn"), iv.wspd.values)
        comps.append(m.component(encode(te, feat, "bqn")))
    return comps


def audit_bqn(tag, df, design, s, feat, cfg, kw, parquet, wcol):
    flist = F0 if feat == "F0" else F1
    tr, iv, te = fold_split(df, design, s, inner_all, early, late)
    if feat == "F1":
        fill_train(tr, [iv, te])
    vc = tr.wspd.value_counts()
    train_atoms = vc[vc >= 50].index.values.astype(float)
    y = te.wspd.values
    cache = OUTD / f"comps_{tag}.npz"
    if cache.exists():
        arr = np.load(cache)
        comps = [__import__("e10_core").BqnComponent(arr[f"c{i}"])
                 for i in range(len(arr.files))]
        t_fit = -1.0
        print(f"[cache] {tag} 组件复用", flush=True)
    else:
        t0 = time.time()
        comps = bqn_comps(tr, iv, te, flist, cfg, kw)
        t_fit = round(time.time() - t0, 1)
        np.savez(cache, **{f"c{i}": c.coef for i, c in enumerate(comps)})
    prod = chunked_mixture_scores(comps, y, chunk=1000,
                                  train_values=train_atoms)
    ref = mixture_adaptive_scores(comps, y, M=64, check=True)
    d = np.abs(prod.crps.values - ref["crps"])
    # 复现检查(同键对齐交付 parquet; 列名随设计不同)
    pq = pd.read_parquet(parquet)
    qc = ("bqn_q50" if "bqn_q50" in pq.columns else
          "BQN_q50" if "BQN_q50" in pq.columns else "q50")
    cc = ("bqn_crps" if "bqn_crps" in pq.columns else
          "BQN_crps" if "BQN_crps" in pq.columns else "crps")
    mine = te[KEYS].copy()
    mine["q50_new"] = ref["q50"]
    mine["crps_new"] = prod.crps.values
    mm = pq.merge(mine, on=KEYS, how="inner")
    repro_q50 = (float(np.abs(mm[qc] - mm.q50_new).max())
                 if mm[qc].notna().any() else None)
    repro_crps = (float(np.abs(mm[cc] - mm.crps_new).max())
                  if mm[cc].notna().any() else None)
    w = te[[c for c in KEYS]] if False else None
    row = dict(
        tag=tag, design=design, station=s, feat=feat, n_rows=len(y),
        n_train=len(tr), cfg=str(cfg), fit_s=t_fit,
        width80_min=round(float(np.nanmin(prod.width80)), 4),
        width80_max=round(float(np.nanmax(prod.width80)), 4),
        prod_vs_ref_max=float(d.max()),
        prod_vs_ref_p99=float(np.percentile(d, 99)),
        prod_vs_ref_median=float(np.median(d)),
        exceed_1e4=int((d > THR).sum()),
        self_conv_max=float(ref["self_conv"].max()),
        self_conv_rel_max=float((ref["self_conv"]
                                 / np.maximum(np.abs(ref["crps"]), 1e-9)
                                 ).max()),
        q50_prod_vs_ref_max=float(np.abs(prod.q50.values - ref["q50"]).max()),
        width80_prod_vs_ref_max=float(np.abs(prod.width80.values
                                            - ref["width80"]).max()),
        repro_q50_max=repro_q50, repro_crps_max=repro_crps)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    with open(OUTD / "entries.jsonl", "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row, te, prod, ref


# ---------------- 数据与选站 ----------------
sel = json.loads((BASE / "T/search_results.json").read_text())["selected"]
cfg_T = dict(degree=int(sel["BQN"]["degree"]),
             width=eval(sel["BQN"]["width"]), lr=float(sel["BQN"]["lr"]))
kw_T = {}
rows_out, entries = [], []

early = late = inner_all = None
dfF0 = load("F0")
dfF1 = load("F1")
stations = sorted(dfF0.station.unique())
inner_all = [st for st in stations if station_hash20(st)]
early = dfF1.valid_time <= pd.Timestamp("2021-08-31 23:59")
late = dfF1.valid_time >= pd.Timestamp("2021-10-01 00:00")


def pick_stations(design_dir, design_name, wcol, prefix):
    dfp = pd.read_parquet(BASE / design_dir / f"{design_dir}_bqn.parquet")
    w = dfp.groupby("station")[wcol].mean().dropna()
    w = w[~w.index.str.contains("Nico")]
    return [w.idxmin(), w.idxmax()]


jobs = []
jobs.append(("T-official", "T", None, "F0", dfF0, cfg_T, kw_T,
             BASE / "T/T_mixture.parquet"))
jobs.append(("T1-terrain", "T", None, "F1", dfF1, cfg_T, kw_T,
             BASE / "T1/T1_bqn.parquet"))
for dn, feat, dfv in (("S", "F0", dfF0), ("S1", "F1", dfF1),
                      ("F", "F0", dfF0), ("F1", "F1", dfF1)):
    wcol = "bqn_width80" if dn in ("S", "F") else "width80"
    for i, st in enumerate(pick_stations(dn, dn, wcol, None)):
        jobs.append((f"{dn}-{'narrow' if i == 0 else 'wide'}", dn[0], st,
                     feat, dfv, dict(degree=12, width=(64, 32), lr=1e-3),
                     dict(val_sub=1000, eval_every=2),
                     BASE / f"{dn}/{dn}_bqn.parquet"))

for tag, design, st, feat, dfv, cfg, kw, pq in jobs:
    row, te, prod, ref = audit_bqn(tag, dfv, design, st, feat, cfg, kw, pq,
                                   None)
    entries.append(row)
    rr = te[KEYS].copy()
    rr["tag"] = tag
    for k in ("crps", "q10", "q50", "q90", "width80"):
        rr[f"prod_{k}"] = prod[k].values
        rr[f"ref_{k}"] = ref[k] if k != "crps" else ref["crps"]
    rr["abs_diff_crps"] = np.abs(prod.crps.values - ref["crps"])
    rr["self_conv"] = ref["self_conv"]
    rows_out.append(rr)

# Nico 登记(S1/F1 已全量自适应交付)
nico_reg = {}
for dn in ("S1", "F1"):
    f = BASE / dn / "V3-07_nico_full_summary.json"
    if f.exists():
        sj = json.loads(f.read_text())
        nico_reg[dn] = dict(n_rows=sj["n_rows"], crps_mean=sj["crps_mean"],
                            self_conv_max=sj["self_conv_max"],
                            self_conv_rel_max=sj["self_conv_rel_max"],
                            source=str(f.relative_to(BASE)))

# QRF/LGBM 确定性抽查(S 设计, 中位 width80 站)
det = {}
try:
    dfp = pd.read_parquet(BASE / "S/S_bqn.parquet")
    w = dfp.groupby("station")["bqn_width80"].mean().dropna()
    st_med = w.sort_values().index[len(w) // 2]
    tr, iv, te = fold_split(dfF0, "S", st_med, inner_all, early, late)
    vc = tr.wspd.value_counts()
    ta = vc[vc >= 50].index.values.astype(float)
    y = te.wspd.values
    comps_q = [QRF(seed=sd, n_jobs=24, min_samples_leaf=50,
                   max_features=1.0).fit(encode(tr, F0, "qrf"),
                                         tr.wspd.values).component(
                                             encode(te, F0, "qrf"))
               for sd in SEEDS]
    pq_q = pd.read_parquet(BASE / "S/S_qrf.parquet")
    pr_q = chunked_mixture_scores(comps_q, y, chunk=1000, train_values=ta)
    mine = te[KEYS].copy()
    mine["q50_new"] = pr_q.q50.values
    mm = pq_q.merge(mine, on=KEYS, how="inner")
    det["qrf_S"] = dict(station=st_med, n_rows=len(y),
                        repro_q50_max=float(np.abs(mm.q50 - mm.q50_new)
                                            .max()))
    comps_l = [LGBM22(seed=sd, n_estimators=300, early_stopping=None,
                      num_leaves=31, min_data_in_leaf=50,
                      learning_rate=0.05).fit(tr[F0], tr.wspd.values
                                              ).component(te[F0])
               for sd in SEEDS]
    pq_l = pd.read_parquet(BASE / "S/S_lgbm.parquet")
    pr_l = chunked_mixture_scores(comps_l, y, chunk=1000, train_values=ta)
    mine = te[KEYS].copy()
    mine["q50_new"] = pr_l.q50.values
    mm = pq_l.merge(mine, on=KEYS, how="inner")
    det["lgbm_S"] = dict(station=st_med, n_rows=len(y),
                         repro_q50_max=float(np.abs(mm.q50 - mm.q50_new)
                                             .max()))
except Exception as e:                                  # noqa
    det["error"] = f"{type(e).__name__}:{e}"

pd.concat(rows_out, ignore_index=True).to_csv(OUTD / "audit_rows.csv",
                                              index=False)
commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                        text=True, cwd="/home/vhyang/zcode/awf").stdout.strip()
entries_all = [json.loads(l) for l in
               open(OUTD / "entries.jsonl").read().splitlines() if l]
summary = dict(
    run="AWF_E10_V3-03_R2", utc=datetime.now(timezone.utc).isoformat(),
    git_commit=commit,
    reference_definition="e10_adaptive.mixture_adaptive_scores M=64: "
    "panel-Simpson E_ij(分位像边界+head/tail 闭式修正) + 闭式 Bernstein "
    "矩 J_ii + τ域 GL E|X-y|; 自收敛=M翻倍",
    sample_selection="T/T1 官方配置全测试行; S/S1/F/F1 每设计 width80 "
    "最窄+最宽站(排除 Nico)整折; Nico 见 nico_entries; 阈值 1e-4",
    input_identity="重训种子/配置/划分与交付运行一致(见各条 cfg 与 "
    "repro_*_max 确定性证据)",
    entries=entries_all, nico_entries=nico_reg, qrf_lgbm_determinism=det,
    exceed_total=int(sum(e["exceed_1e4"] for e in entries_all)),
    rows_total=int(sum(e["n_rows"] for e in entries_all)))
(OUTD / "summary.json").write_text(json.dumps(summary, ensure_ascii=False,
                                              indent=2))
print(json.dumps({k: summary[k] for k in ("exceed_total", "rows_total")},
                 ensure_ascii=False))
