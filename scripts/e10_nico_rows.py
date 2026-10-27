#!/usr/bin/env python
"""AWF E10 — R07 两组剩余缺口的证据导出

组2(真实 Nico 行受控精度): 确定性重训 Nico 折落盘三分量系数, 抽真实
  极端风险行(S1/F1 各5行: crps min/median/max+随机2), 记录键/obs/系数/
  生产评分(M=64, 即交付值)/加密参照(M=256)/独立恒等式参照, 并给出对
  全站汇总显示精度的影响。
组1(身份对应): 另由 build 阶段生成对应表(见 28 号包构建脚本)。

用法: python e10_nico_rows.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/vhyang/zcode/awf/scripts")
from e10_adaptive import (J_analytic, E_cross, mixture_crps_only,
                          mixture_adaptive_scores)
from e10_models import BQN, encode
from e10_protocol import station_hash20

DATA = Path("/mnt/d/data/awfdata")
REMAF = Path("/home/vhyang/zcode/awf/outputs/e5_baseline/rema/"
             "rema_features_by_station.csv")
TFV3 = Path("/home/vhyang/zcode/awf/outputs/review_v3/terrain_features_v3.csv")
B = Path("/home/vhyang/zcode/awf/outputs/bqn_lgbm_qrf")
OUTD = B / "V3-03_risk_coverage"
OUTD.mkdir(exist_ok=True)
SEEDS = [17, 42, 2026]
KEYS = ["station", "init", "valid_time", "lead_h"]
F0 = ["u10", "v10", "fcst_spd", "t2m", "prmsl", "rh2m", "lead_h",
      "month_sin", "month_cos", "lat", "lon", "elevation", "terrain"]
F1E = ["slope1k", "slope5k", "slope25k", "prominence", "fetch_drop",
       "katab_align_EN", "is_flat", "is_missing"]
F1 = F0 + F1E


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
    st = df.station.map(tf.status)
    df["is_flat"] = (st == "flat").astype(int).values
    df["is_missing"] = (st == "missing").astype(int).values
    df.loc[df.is_flat == 1, "katab_align_EN"] = 0.0
    df.loc[df.is_missing == 1, "katab_align_EN"] = 0.0
    return df


df = load()
stations = sorted(df.station.unique())
inner_all = [s for s in stations if station_hash20(s)]
early = df.valid_time <= pd.Timestamp("2021-08-31 23:59")
late = df.valid_time >= pd.Timestamp("2021-10-01 00:00")

cache = OUTD / "comps_Nico_full.npz"
if cache.exists():
    arr = np.load(cache)
    coef_S1 = {"coef": arr["S1"]}
    coef_F1 = {"coef": arr["F1"]}
    print("[cache] Nico 组件复用")
else:
    coef_S1, coef_F1 = {}, {}
    for design in ("S", "F"):
        s = "Nico"
        if design == "S":
            test = df.station == s
            inner = (~test) & df.station.isin(
                [c for c in inner_all if c != s])
            tr = df[(~test) & (~inner)].reset_index(drop=True)
            iv = df[inner].reset_index(drop=True)
            te = df[test].reset_index(drop=True)
        else:
            test = (df.station == s) & late
            inner = ((df.station != s) & early & df.station.isin(
                [c for c in inner_all if c != s])
                & (df.valid_time.dt.month == 8))
            tr = df[(df.station != s) & early & (~inner)].reset_index(
                drop=True)
            iv = df[inner].reset_index(drop=True)
            te = df[test].reset_index(drop=True)
        for c in F1E[:5]:
            med = float(tr[c].median())
            for d in (tr, iv, te):
                d[c] = d[c].fillna(med)
        cs = []
        for sd in SEEDS:
            t0 = time.time()
            m = BQN(seed=sd, degree=12, width=(64, 32), lr=1e-3,
                    val_sub=1000, eval_every=2).fit(
                encode(tr, F1, "bqn"), tr.wspd.values,
                encode(iv, F1, "bqn"), iv.wspd.values)
            cs.append(m.component(encode(te, F1, "bqn")).coef)
            print(f"  {design}1 seed {sd}: coef_max={cs[-1].max():.3e} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        (coef_S1 if design == "S" else coef_F1)["coef"] = np.stack(cs)
        (coef_S1 if design == "S" else coef_F1)["te"] = te
    np.savez(cache, S1=coef_S1["coef"], F1=coef_F1["coef"])
    print("[saved]", cache)

from e10_core import BqnComponent
results = {}
for design in ("S1", "F1"):
    d1 = design
    te = None
    # 重建 te(划分确定性): 重读原逻辑
    s = "Nico"
    if design == "S1":
        test = df.station == s
        inner = (~test) & df.station.isin([c for c in inner_all if c != s])
        te = df[test].reset_index(drop=True)
    else:
        test = (df.station == s) & late
        te = df[test].reset_index(drop=True)
    coef = coef_S1["coef"] if design == "S1" else coef_F1["coef"]
    comps = [BqnComponent(coef[i]) for i in range(3)]
    y = te.wspd.values

    # 1) Nico 折确定性: 重训分数 vs 交付 CSV
    delivered = pd.read_csv(B / f"{d1}/V3-07_nico_full_scores.csv")
    prod = mixture_crps_only(comps, y, M=64)
    det = float(np.abs(prod - delivered.crps.values).max())
    det_rel = float((np.abs(prod - delivered.crps.values)
                     / delivered.crps.values).max())
    print(f"[{d1}] 重训 vs 交付CSV: max_abs={det:.3e} max_rel={det_rel:.3e}")

    # 2) 抽 5 行: crps min/median/max + 随机2(固定种子)
    order = np.argsort(prod)
    rng = np.random.default_rng(27)
    pick = [int(order[0]), int(order[len(order) // 2]), int(order[-1]),
            int(rng.integers(len(order))), int(rng.integers(len(order)))]
    rows = []
    for r in pick:
        cs_r = [BqnComponent(coef[i][r:r + 1]) for i in range(3)]
        yv = np.array([y[r]])
        m64 = mixture_crps_only(cs_r, yv, M=64)[0]
        m256 = mixture_crps_only(cs_r, yv, M=256)[0]
        # 独立恒等式参照(与生产/自适应无共享路径): 成对 E 用
        # E|Xi-Xj|=E[Xi]+E[Xj]-2E[min] 的解析均值 + E[min] 稠密梯形
        def E_idref(ci, cj, n_tau=200001, n_x=400000):
            EXi, EXj = float(ci.coef.mean()), float(cj.coef.mean())
            hi = max(float(ci.coef[:, -1].max()),
                     float(cj.coef[:, -1].max())) * 1.0001
            x = np.unique(np.concatenate([
                np.linspace(0, 50, 20000),
                np.geomspace(50, max(hi, 51), n_x)]))
            F = []
            for c in (ci, cj):
                t = np.linspace(0, 1, n_tau)
                xq = c.Q_of(t)[0]
                F.append(np.interp(x, xq, t, left=0.0, right=1.0))
            Emin = np.trapezoid((1 - F[0]) * (1 - F[1]), x)
            return EXi + EXj - 2 * Emin
        Ey = float(np.mean([c.Ey(yv)[0] for c in cs_r]))
        E = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                E[i, j] = 2 * float(J_analytic(cs_r[i])[0]) if i == j \
                    else E_idref(cs_r[i], cs_r[j])
        ref = Ey - 0.5 * E.mean()
        rows.append(dict(
            row_index=int(r),
            key=dict(station="Nico", init=str(te.init.values[r]),
                     valid_time=str(te.valid_time.values[r]),
                     lead_h=float(te.lead_h.values[r]), obs=float(y[r])),
            coef_seed17=coef[0][r].tolist(), coef_seed42=coef[1][r].tolist(),
            coef_seed2026=coef[2][r].tolist(), weights=[1 / 3] * 3,
            prod_M64=float(delivered.crps.values[r]),
            retrain_M64=float(m64), refined_M256=float(m256),
            identity_ref=float(ref),
            abs_refined_vs_prod=abs(float(m256) - float(m64)),
            rel_refined_vs_prod=abs(float(m256) - float(m64)) / abs(float(m64)),
            rel_identity_vs_refined=abs(ref - m256) / max(abs(m256), 1e-9)))
        print(f"  row {r}: prod={m64:.6e} M256={m256:.6e} "
              f"ref={ref:.6e} rel(M256-prod)={rows[-1]['rel_refined_vs_prod']:.2e}")
    results[design] = dict(n_rows_total=len(y), determinism_vs_delivered=dict(
        max_abs=det, max_rel=det_rel), sampled_rows=rows)

json.dump(results, open(OUTD / "nico_real_rows_precision.json", "w"),
          ensure_ascii=False, indent=2)
print("written", OUTD / "nico_real_rows_precision.json")
