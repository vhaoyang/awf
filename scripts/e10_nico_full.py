#!/usr/bin/env python
"""AWF E10 — V3-07 收口: Nico 折(S1/F1)完整重训 + 全量行自适应评分 + 回填

R05 裁决: 50 行抽样不能代替全站交付。本脚本:
  1) 按 e10_run_F1.py 的 fold-34 划分(S/F 两分支)精确重建 Nico 折;
  2) 3 种子重训 BQN(预设 degree=12, width=(64,32), lr=1e-3,
     val_sub=1000, eval_every=2)——_val_crps 已加无网格回退,
     极端支撑下早停正常工作(此前该折在验证评分阶段即失败);
  3) 全部测试行(S1: 1502, F1: 447)用 e10_adaptive 无网格混合评分;
  4) 回填 {S1,F1}_bqn.parquet 的 Nico 行与 folds CSV;
  5) 旧(成功子集)汇总先另存为 *_subset_backup.json 作敏感性对照。

用法: python e10_nico_full.py S|F
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
from e10_models import BQN, encode
from e10_protocol import station_hash20

DATA = Path("/mnt/d/data/awfdata")
REMAF = Path("/home/vhyang/zcode/awf/outputs/e5_baseline/rema/"
             "rema_features_by_station.csv")
TFV3 = Path("/home/vhyang/zcode/awf/outputs/review_v3/terrain_features_v3.csv")
BASE = Path("/home/vhyang/zcode/awf/outputs/bqn_lgbm_qrf")
SEEDS = [17, 42, 2026]
F0 = ["u10", "v10", "fcst_spd", "t2m", "prmsl", "rh2m", "lead_h",
      "month_sin", "month_cos", "lat", "lon", "elevation", "terrain"]
F1_EXTRA = ["slope1k", "slope5k", "slope25k", "prominence", "fetch_drop",
            "katab_align_EN", "is_flat", "is_missing"]
F1 = F0 + F1_EXTRA
DESIGN = sys.argv[1].upper()
OUT = BASE / f"{DESIGN}1"
KEYS = ["station", "init", "valid_time", "lead_h"]


def load():
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


df = load()
stations = sorted(df.station.unique())
inner_all = [s for s in stations if station_hash20(s)]
early = df.valid_time <= pd.Timestamp("2021-08-31 23:59")
late = df.valid_time >= pd.Timestamp("2021-10-01 00:00")

s = "Nico"
if DESIGN == "S":
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
med = fill_train(tr, [iv, te])
assert len(te) in (1502, 447), f"Nico 测试行数异常: {len(te)}"

print(f"[{DESIGN}1-Nico] n_train={len(tr)} n_inner={len(iv)} n_test={len(te)}"
      f" fill_med={ {k: round(v,3) for k,v in med.items()} }", flush=True)

comps, seed_info = [], []
for sd in SEEDS:
    t0 = time.time()
    m = BQN(seed=sd, degree=12, width=(64, 32), lr=1e-3, val_sub=1000,
            eval_every=2).fit(encode(tr, F1, "bqn"), tr.wspd.values,
                              encode(iv, F1, "bqn"), iv.wspd.values)
    comp = m.component(encode(te, F1, "bqn"))
    comps.append(comp)
    seed_info.append(dict(
        seed=sd, best_val_crps=(round(float(m.best_val_crps), 6)
                                if np.isfinite(m.best_val_crps) else None),
        n_val_evals=len(m.history),
        best_eval_idx=int(np.argmin(m.history)) if m.history else None,
        elapsed_s=round(time.time() - t0, 1),
        coef_max=float(comp.coef.max()), coef_min=float(comp.coef.min()),
        finite=bool(np.isfinite(comp.coef).all()),
        monotone=bool((np.diff(comp.coef, axis=1) >= -1e-9).all())))
    print(f"  seed {sd}: best_val_crps={seed_info[-1]['best_val_crps']} "
          f"evals={len(m.history)} coef_max={comp.coef.max():.3e} "
          f"({seed_info[-1]['elapsed_s']}s)", flush=True)

y = te.wspd.values
t0 = time.time()
res = mixture_adaptive_scores(comps, y, M=64, check=True)
print(f"  adaptive scoring {len(y)} rows: {time.time()-t0:.1f}s, "
      f"self_conv max={res['self_conv'].max():.3e}", flush=True)

cov80 = float(((y >= res["q10"]) & (y <= res["q90"])).mean())
mae = float(np.abs(res["q50"] - y).mean())
scores = pd.DataFrame(dict(
    station="Nico", init=te.init.values, valid_time=te.valid_time.values,
    lead_h=te.lead_h.values, obs=y, crps=res["crps"], q10=res["q10"],
    q50=res["q50"], q90=res["q90"], p_gt15=res["p_gt15"],
    width80=res["width80"], self_conv=res["self_conv"]))
scores.to_csv(OUT / "V3-07_nico_full_scores.csv", index=False)

commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                        text=True, cwd="/home/vhyang/zcode/awf").stdout.strip()
summary = dict(
    run="AWF_E10_V3-07_R2", design=f"{DESIGN}1", station="Nico",
    n_rows=int(len(y)), n_train=int(len(tr)), n_inner=int(len(iv)),
    method="adaptive panel-Simpson + closed-form Bernstein moments "
           "(e10_adaptive, M=64, check=2M)",
    seeds=seed_info,
    crps_mean=float(np.mean(res["crps"])), crps_min=float(np.min(res["crps"])),
    crps_max=float(np.max(res["crps"])), mae=mae, cov80=cov80,
    q50_mean=float(np.mean(res["q50"])),
    q10_mean=float(np.mean(res["q10"])), q90_mean=float(np.mean(res["q90"])),
    width80_mean=float(np.mean(res["width80"])),
    p_gt15_mean=float(np.mean(res["p_gt15"])),
    self_conv_max=float(res["self_conv"].max()),
    self_conv_rel_max=float((res["self_conv"]
                             / np.maximum(np.abs(res["crps"]), 1e-9)).max()),
    note="V3-07_nico_full_scores.csv 为逐行正式评分; 本 JSON 为折级摘要",
    git_commit=commit,
    utc=datetime.now(timezone.utc).isoformat())
(OUT / "V3-07_nico_full_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps({k: summary[k] for k in
                  ("n_rows", "crps_mean", "crps_min", "crps_max", "mae",
                   "cov80", "q50_mean", "self_conv_max")}, indent=1))

# ---- 回填 parquet 与 folds CSV(先备份) ----
pq = OUT / f"{DESIGN}1_bqn.parquet"
old = pd.read_parquet(pq)
shutil.copy2(pq, OUT / f"{DESIGN}1_bqn.parquet.bak_pre_v307full")
mask = old.station == "Nico"
assert old.loc[mask, "crps"].isna().all(), "Nico 行已有评分, 拒绝覆盖"
key = scores[KEYS + ["crps", "q10", "q50", "q90", "p_gt15", "width80"]]
merged = old.merge(key, on=KEYS, how="left", suffixes=("", "_new"))
for c in ("crps", "q10", "q50", "q90", "p_gt15", "width80"):
    merged[c] = merged[c].fillna(merged.pop(f"{c}_new"))
assert len(merged) == len(old)
merged = merged[old.columns]
merged.to_parquet(pq, index=False)

folds_f = OUT / f"{DESIGN}1_bqn_folds.csv"
shutil.copy2(folds_f, OUT / f"{DESIGN}1_bqn_folds.csv.bak_pre_v307full")
folds = pd.read_csv(folds_f)
fm = folds.station == "Nico"
assert fm.sum() == 1
folds.loc[fm, "status"] = "ok-adaptive"
folds.loc[fm, "crps"] = round(float(res["crps"].mean()), 5)
folds.loc[fm, "mae"] = round(mae, 5)
folds.to_csv(folds_f, index=False)

# ---- 旧成功子集汇总备份(敏感性对照) ----
d1 = f"{DESIGN}1"
for f in ("summary_v2.json", f"{d1}_per_station.csv",
          f"{d1}_bootstrap.csv"):
    src = OUT / f
    if src.exists():
        shutil.copy2(src, OUT / f"{f}.subset_backup")
print(f"[{d1}-Nico] parquet/folds 回填完成, 旧汇总已备份", flush=True)
