#!/usr/bin/env python
"""AWF E10 — F1 地形消融：F0→F1 对照（模型结构冻结, 方案 §二）

运行号: AWF_E10_F1_R1（每个 design×model 一个子运行）
F1 = F0 的 13 列 + 5 个冻结 REMA 标量 + 对齐度 + 2 状态 = 21 列
（缺测以训练期中位数填充, 协议 feature_manifest）。选中配置/preset 与
F0 完全一致; 输出 outputs/bqn_lgbm_qrf/{T1,S1,F1}/ 与 F0 逐行同键可直接合并。

用法: python e10_run_F1.py <design T|S|F> <model bqn|qrf|lgbm> [resume]
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/vhyang/zcode/awf/scripts")
from e10_models import (BQN, LGBM22, QRF, chunked_mixture_scores, encode)
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
DESIGN = sys.argv[1].upper()
MODEL = sys.argv[2]
RESUME = len(sys.argv) > 3 and sys.argv[3] == "resume"
OUT = BASE / f"{DESIGN}1"
OUT.mkdir(parents=True, exist_ok=True)


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
    """REMA 标量缺测以训练期中位数填充(其余表不动)。"""
    med = {}
    for c in F1_EXTRA[:5]:
        med[c] = float(df_tr[c].median())
    for d in [df_tr] + df_other_list:
        for c, v in med.items():
            d[c] = d[c].fillna(v)
    return med


F1 = F0 + F1_EXTRA
df = load()
stations = sorted(df.station.unique())
inner_all = [s for s in stations if station_hash20(s)]
vc = df.wspd.value_counts()
train_atoms = vc[vc >= 50].index.values.astype(float)

sel = {"T": json.loads((BASE / "T/search_results.json").read_text())
       ["selected"], "S": None, "F": None}[DESIGN]

parts, fold_rows = [], []
done_folds = set()
fname = f"{DESIGN}1_{MODEL}.parquet"
if RESUME and (OUT / fname).exists():
    old = pd.read_parquet(OUT / fname)
    done_folds = set(old.fold.unique())
    parts = [old]
    print(f"[resume] {len(done_folds)} folds", flush=True)

early = df.valid_time <= pd.Timestamp("2021-08-31 23:59")
late = df.valid_time >= pd.Timestamp("2021-10-01 00:00")

if DESIGN == "T":
    folds = [("T", None)]
else:
    folds = [(i, s) for i, s in enumerate(stations)]

t00 = time.time()
for fi, (i, s) in enumerate(folds):
    if i in done_folds:
        continue
    if DESIGN == "T":
        tr = df[df.month <= 8].reset_index(drop=True)
        iv = df[df.month == 9].reset_index(drop=True)
        te = df[df.month >= 10].reset_index(drop=True)
    elif DESIGN == "S":
        test = df.station == s
        inner_st = [c for c in inner_all if c != s]
        inner = test.__invert__() & df.station.isin(inner_st)
        tr = df[test.__invert__() & ~inner].reset_index(drop=True)
        iv = df[inner].reset_index(drop=True)
        te = df[test].reset_index(drop=True)
    else:
        test = (df.station == s) & late
        if int(test.sum()) == 0:
            continue
        inner = ((df.station != s) & early & df.station.isin(
            [c for c in inner_all if c != s])
            & (df.valid_time.dt.month == 8))
        tr = df[(df.station != s) & early & ~inner].reset_index(drop=True)
        iv = df[inner].reset_index(drop=True)
        te = df[test].reset_index(drop=True)
    med = fill_train(tr, [iv, te])
    row = dict(fold=i, station=s, n_train=len(tr), n_test=len(te),
               fill_med={k: round(v, 3) for k, v in med.items()})
    t0 = time.time()
    fail = None
    comps = []
    try:
        for sd in SEEDS:
            if MODEL == "bqn":
                # 15-02: T 从冻结搜索读取; S/F 恢复各自 F0 预设(留站隔离)
                if DESIGN == "T":
                    _sel = json.loads(
                        (BASE / "T/search_results.json").read_text()
                    )["selected"]["BQN"]
                    cfg = dict(degree=int(_sel["degree"]),
                               width=eval(_sel["width"]),
                               lr=float(_sel["lr"]))
                    kw = {}
                else:
                    # S/F 预设: 与 F0 的 e10_run_S/F.py 完全一致
                    cfg = dict(degree=12, width=(64, 32), lr=1e-3)
                    kw = dict(val_sub=1000, eval_every=2)
                m = BQN(seed=sd, **cfg, **kw).fit(
                    encode(tr, F1, "bqn"), tr.wspd.values,
                    encode(iv, F1, "bqn"), iv.wspd.values)
                comps.append(m.component(encode(te, F1, "bqn")))
            elif MODEL == "qrf":
                cfg = dict(min_samples_leaf=10, max_features=0.7) \
                    if DESIGN == "T" else dict(min_samples_leaf=50,
                                               max_features=1.0)
                m = QRF(seed=sd, n_jobs=24, **cfg).fit(
                    encode(tr, F1, "qrf"), tr.wspd.values)
                comps.append(m.component(encode(te, F1, "qrf")))
            else:
                ne = 1000 if DESIGN == "T" else 300
                es = ((iv[F1], iv.wspd.values) if DESIGN == "T" else None)
                m = LGBM22(seed=sd, n_estimators=ne, early_stopping=es,
                           num_leaves=31, min_data_in_leaf=100
                           if DESIGN == "T" else 50,
                           learning_rate=0.05).fit(tr[F1], tr.wspd.values)
                comps.append(m.component(te[F1]))
        preds = chunked_mixture_scores(comps, te.wspd.values, chunk=1000,
                                       train_values=train_atoms)
    except Exception as e:                                    # noqa
        fail = f"{type(e).__name__}:{e}"
        preds = pd.DataFrame({k: np.full(len(te), np.nan) for k in
                              ("crps", "q10", "q50", "q90", "p_gt15",
                               "width80", "_grid_check")})
    row["status"] = "ok" if fail is None else fail
    row["crps"] = (round(float(preds.crps.mean()), 5)
                   if fail is None else None)
    row["mae"] = (round(float(np.abs(preds.q50 - te.wspd.values).mean()), 5)
                  if fail is None else None)
    row["elapsed_s"] = round(time.time() - t0, 1)
    fold_rows.append(row)
    base = te[["station", "init", "valid_time", "lead_h", "wspd",
               "fcst_spd"]].rename(columns={"wspd": "obs",
                                            "fcst_spd": "gfs"}).copy()
    base["fold"] = i
    parts.append(pd.concat([base.reset_index(drop=True),
                            preds.reset_index(drop=True)], axis=1))
    print(f"[{DESIGN}1-{MODEL} {fi + 1}/{len(folds)}] {s}: "
          f"{row['crps']} ({row['elapsed_s']}s) "
          f"{'FAIL ' + fail[:40] if fail else ''}", flush=True)
    pd.concat(parts, ignore_index=True).to_parquet(OUT / fname,
                                                   index=False)
pd.DataFrame(fold_rows).to_csv(OUT / f"{DESIGN}1_{MODEL}_folds.csv",
                               index=False)
ok = pd.DataFrame(fold_rows)
print(json.dumps(dict(run=f"AWF_E10_F1_R1 {DESIGN} {MODEL}",
                      folds_ok=int((ok.status == "ok").sum()),
                      folds_fail=int((ok.status != "ok").sum()),
                      pooled_crps=round(float(ok.crps.dropna().mean()), 4)
                      if ok.crps.notna().any() else None,
                      utc=datetime.now(timezone.utc).isoformat(),
                      elapsed=round(time.time() - t00, 1)), indent=1))
