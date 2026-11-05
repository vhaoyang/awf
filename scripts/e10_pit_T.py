#!/usr/bin/env python
"""Refit temporal components and construct atom-aware randomized PIT values.

U = F_mix(y-) + V * (F_mix(y)-F_mix(y-)), with a fixed random seed for V.
The program first compares refitted predictions with archived keyed values
and stops if the identity check fails. Ten equal-width bins are descriptive;
they are not a formal calibration test. Fitting and scoring are performed
when this program is run."""
from ast import literal_eval
import os
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e10_core import BqnComponent, LgbmComponent, QrfComponent
from e10_models import BQN, LGBM22, QRF, chunked_mixture_scores, encode

DATA = Path(os.environ.get("AWF_DATA_ROOT", "data"))
B = Path("outputs/bqn_lgbm_qrf")
OUTD = Path("outputs/manuscript_ams/pit")
OUTD.mkdir(parents=True, exist_ok=True)
F0 = ["u10", "v10", "fcst_spd", "t2m", "prmsl", "rh2m", "lead_h",
      "month_sin", "month_cos", "lat", "lon", "elevation", "terrain"]
SEEDS = [17, 42, 2026]
KEYS = ["station", "init", "valid_time", "lead_h"]

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
tr = df[df.month <= 8].reset_index(drop=True)
va = df[df.month == 9].reset_index(drop=True)
te = df[df.month >= 10].reset_index(drop=True)
Xte_b, Xte_q = encode(te, F0, "bqn"), encode(te, F0, "qrf")
Xtr_b, Xtr_q = encode(tr, F0, "bqn"), encode(tr, F0, "qrf")
Xva_b = encode(va, F0, "bqn")
y = te.wspd.values
print(f"T: train={len(tr)} val={len(va)} test={len(te)}", flush=True)

sel = json.loads((B / "T/search_results.json").read_text())["selected"]
t0 = time.time()
comps = {"BQN": [], "QRF": [], "LGBM": []}
for sd in SEEDS:
    comps["BQN"].append(BQN(
        seed=sd, degree=int(sel["BQN"]["degree"]),
        width=literal_eval(sel["BQN"]["width"]), lr=float(sel["BQN"]["lr"])
    ).fit(Xtr_b, tr.wspd.values, Xva_b, va.wspd.values).component(Xte_b))
    print(f"BQN seed {sd} ({time.time()-t0:.0f}s)", flush=True)
for sd in SEEDS:
    comps["QRF"].append(QRF(seed=sd, n_jobs=24, **sel["QRF"]).fit(
        Xtr_q, tr.wspd.values).component(Xte_q))
    print(f"QRF seed {sd} done", flush=True)
for sd in SEEDS:
    comps["LGBM"].append(LGBM22(
        seed=sd, n_estimators=1000,
        early_stopping=(va[F0], va.wspd.values),
        num_leaves=31, min_data_in_leaf=100, learning_rate=0.05
    ).fit(tr[F0], tr.wspd.values).component(te[F0]))
    print(f"LGBM seed {sd} done", flush=True)

# ---------- 身份验证(冻结逐键比对) ----------
frozen = pd.read_parquet(B / "T/T_mixture.parquet")
report, ok_all = {}, True
for mdl in ("BQN", "QRF", "LGBM"):
    pred = chunked_mixture_scores(comps[mdl], y, chunk=1500,
                                  train_values=np.array([]))
    tmp = te[KEYS].copy()
    tmp["q50_new"] = pred.q50.values
    tmp["crps_new"] = pred.crps.values
    mm = frozen.merge(tmp, on=KEYS, how="inner")
    dq = float((mm[f"{mdl}_q50"] - mm.q50_new).abs().max())
    rel = float(((mm[f"{mdl}_crps"] - mm.crps_new).abs()
                 / mm[f"{mdl}_crps"].clip(lower=1e-6)).max())
    ok = dq < 1e-4 and rel < 1e-4
    ok_all &= ok
    report[mdl] = dict(max_abs_q50=dq, max_rel_crps=rel,
                       n=int(len(mm)), ok=bool(ok))
    print(f"[身份] {mdl}: q50 {dq:.2e} rel_crps {rel:.2e} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
if not ok_all:
    (OUTD / "pit_identity_FAIL.json").write_text(json.dumps(report))
    sys.exit("Prediction identity check failed; PIT generation stopped.")


# ---------- 原子感知 F 左右极限 ----------
def Fl_Fr(cs, y):
    n = len(y)
    Fl, Fr = np.zeros(n), np.zeros(n)
    for c in cs:
        if isinstance(c, BqnComponent):
            lo, hi = np.zeros(n), np.ones(n)
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                le = c.Q_diag(mid) < y
                lo, hi = np.where(le, mid, lo), np.where(le, hi, mid)
            f = 0.5 * (lo + hi)
            Fl += f
            Fr += f
        elif isinstance(c, QrfComponent):
            # 分布=(1/T)Σ LeafECDF; 逐行 F = 各叶 searchsorted 均值
            fl = np.zeros(n)
            fr = np.zeros(n)
            for t in range(c.T):
                leaves = c.leaf_test[:, t]
                uniq, inv = np.unique(leaves, return_inverse=True)
                lut = c.tls[t]
                for ui, lid in enumerate(uniq):
                    rows = np.where(inv == ui)[0]
                    s = lut[lid]
                    fl[rows] += np.searchsorted(s, y[rows], "left") / len(s)
                    fr[rows] += np.searchsorted(s, y[rows], "right") / len(s)
            Fl += fl / c.T
            Fr += fr / c.T
        elif isinstance(c, LgbmComponent):
            # 分段线性 Q 的广义逆: 增段线性插值(连续), 节点平段=原子(左右极限=τ跨度)
            t_n, q_n = c.taus, c.Q        # (n, K+2)
            for r in range(n):
                q, yv = q_n[r], y[r]
                if yv > q[-1]:
                    fl_, fr_ = 1.0, 1.0
                elif yv < q[0]:
                    fl_, fr_ = 0.0, 0.0
                else:
                    i0 = np.searchsorted(q, yv, "left")
                    if q[i0] == yv:      # 节点值: 连续到达 + 原子不进左极限
                        i1 = np.searchsorted(q, yv, "right") - 1
                        fl_, fr_ = float(t_n[i0]), float(t_n[i1])
                    else:                 # 严格位于两节点之间: 线性插值
                        k = np.searchsorted(q, yv, "right") - 1
                        span = q[k + 1] - q[k]
                        frac = (yv - q[k]) / span if span > 0 else 0.0
                        fl_ = fr_ = float(
                            t_n[k] + (t_n[k + 1] - t_n[k]) * frac)
                Fl[r] += fl_
                Fr[r] += fr_
    return Fl / len(cs), Fr / len(cs)


rng = np.random.default_rng(42)
V = rng.random(len(y))
pit = {}
FF = {}
for mdl in ("BQN", "QRF", "LGBM"):
    t0 = time.time()
    Fl, Fr = Fl_Fr(comps[mdl], y)
    FF[mdl] = (Fl, Fr)
    U_raw = Fl + V * (Fr - Fl)
    U = np.clip(U_raw, 0.0, 1.0)
    counts, edges = np.histogram(U, bins=np.linspace(0, 1, 11))
    pit[mdl] = dict(counts=counts.tolist(), edges=edges.tolist(),
                    U_min=float(U.min()), U_max=float(U.max()),
                    n_endpoint_0=int((U == 0).sum()),
                    n_endpoint_1=int((U == 1).sum()),
                    sum=int(counts.sum()))
    pit[mdl]["n_preclip_lt0"] = int((U_raw < 0).sum())
    pit[mdl]["n_preclip_gt1"] = int((U_raw > 1).sum())
    pit[mdl]["n_clip_changed"] = int((U != U_raw).sum())
    np.save(OUTD / f"U_{mdl}.npy", U)
    print(f"[PIT] {mdl}: sum={counts.sum()} U[{U.min():.4f},{U.max():.4f}] "
          f"端点0/1: {(U==0).sum()}/{(U==1).sum()} ({time.time()-t0:.0f}s)",
          flush=True)

# ---------- Keyed PIT records and consistency checks ----------
tbl = te[KEYS].copy()
tbl["obs_wspd"] = y
for mdl in ("BQN", "QRF", "LGBM"):
    Fl, Fr = FF[mdl]
    np.save(OUTD / f"Fl_Fr_{mdl}.npy", np.column_stack([Fl, Fr]))
    tbl[f"Fl_{mdl}"] = Fl
    tbl[f"Fr_{mdl}"] = Fr
    tbl[f"U_{mdl}"] = np.clip(Fl + V * (Fr - Fl), 0.0, 1.0)
tbl["V_rng42"] = V
tbl.to_parquet(OUTD / "pit_keys.parquet", index=False)
tbl.to_csv(OUTD / "pit_keys.csv.gz", index=False)  # Plain-text copy of the diagnostic table

ev = {}
ev["n_rows"] = int(len(tbl))
ev["n_unique_keys"] = int(tbl[KEYS].drop_duplicates().shape[0])
ev["n_stations"] = int(tbl.station.nunique())
frz = pd.read_parquet(B / "T/T_mixture.parquet")
k1 = tbl[KEYS].astype(str).agg("|".join, axis=1)
k2 = frz[KEYS].astype(str).agg("|".join, axis=1)
import hashlib
ev["keyset_sha256_this"] = hashlib.sha256(
    "".join(sorted(k1)).encode()).hexdigest()
ev["keyset_sha256_frozen"] = hashlib.sha256(
    "".join(sorted(k2)).encode()).hexdigest()
ev["keysets_identical"] = bool(set(k1) == set(k2))
for mdl in ("BQN", "QRF", "LGBM"):
    Fl, Fr = FF[mdl]
    ev[f"{mdl}_finite"] = bool(np.isfinite(Fl).all() and np.isfinite(Fr).all())
    ev[f"{mdl}_viol_0FlFr1"] = int(((Fl < 0) | (Fr > 1) | (Fl > Fr)).sum())
# 代表行: QRF最大原子 / LGBM最大原子 / BQN一例U=1 / LGBM全部U=0行摘要
rep = {}
fl, fr = FF["QRF"]; i = int(np.argmax(fr - fl))
rep["qrf_max_atom"] = {**{k: str(v) for k, v in
                          tbl.iloc[i][KEYS + ["obs_wspd"]].items()},
                       "Fl": float(fl[i]), "Fr": float(fr[i])}
fl, fr = FF["LGBM"]; i = int(np.argmax(fr - fl))
rep["lgbm_max_atom"] = {**{k: str(v) for k, v in
                          tbl.iloc[i][KEYS + ["obs_wspd"]].items()},
                        "Fl": float(fl[i]), "Fr": float(fr[i])}
u = np.clip(FF["BQN"][0] + V * (FF["BQN"][1] - FF["BQN"][0]), 0, 1)
i = int(np.where(u == 1)[0][0])
rep["bqn_U1_example"] = {**{k: str(v) for k, v in
                           tbl.iloc[i][KEYS + ["obs_wspd"]].items()},
                         "Fl": float(FF["BQN"][0][i]), "Fr": float(FF["BQN"][1][i])}
m0 = (np.load(OUTD / "U_LGBM.npy") == 0)
rep["lgbm_U0_rows"] = dict(
    n=int(m0.sum()),
    all_Fl_zero=bool((FF["LGBM"][0][m0] == 0).all()),
    all_Fr_zero=bool((FF["LGBM"][1][m0] == 0).all()),
    max_Fr=float(FF["LGBM"][1][m0].max()),
    obs_all_zero=bool((y[m0] == 0).all()))
ev["representative_rows"] = rep
ev["V_mapping"] = ("V=default_rng(42).random(n) 按本表行序分配; 行序=确定性装载管"
                   "线(读parquet→合并registry→dropna(wspd)→reset_index→month>=10)"
                   "所得 te 顺序, 本表即该顺序, 键-V 映射由本表显式给出")
# ---------- Quantile-node and support diagnostics ----------
nodes = {}
# LGBM: 每种子全行节点数组 (n, K+2) + 平段扫描 + 递增段例
for si, c in enumerate(comps["LGBM"]):
    np.save(OUTD / f"lgbm_nodes_seed{SEEDS[si]}.npy", c.Q)
q = np.stack([c.Q for c in comps["LGBM"]])          # (3, n, K+2)
flat = (np.diff(q, axis=2) == 0).any(axis=2)         # 行内任一相邻节点相等
nodes["lgbm_rows_with_flat_segment_per_seed"] = flat.sum(axis=1).tolist()
nodes["lgbm_rows_with_any_seed_flat"] = int(flat.any(axis=0).sum())
inc_rows = np.where(~flat.any(axis=0))[0]            # 全种子无平段行=递增段
r0 = int(inc_rows[0])
k = 5                                               # 展示第6段(τ0.25→0.30示意)
nodes["lgbm_inc_example"] = dict(
    row=r0, **{f"key_{kk}": str(tbl.iloc[r0][kk]) for kk in KEYS},
    obs=float(y[r0]), seg_tau=[float(comps["LGBM"][0].taus[k]),
                               float(comps["LGBM"][0].taus[k+1])],
    seg_q_per_seed={str(SEEDS[si]): [float(q[si, r0, k]),
                                     float(q[si, r0, k+1])]
                    for si in range(3)},
    mixture_Fl=float(FF["LGBM"][0][r0]), mixture_Fr=float(FF["LGBM"][1][r0]))
if flat.any():
    r1 = int(np.where(flat.any(axis=0))[0][0])
    kk = int(np.where(np.diff(q[:, r1, :], axis=1) == 0)[1][0])
    nodes["lgbm_flat_example"] = dict(
        row=r1, **{f"key_{k2}": str(tbl.iloc[r1][k2]) for k2 in KEYS},
        repeat_q=float(q[0, r1, kk]),
        tau_span=[float(comps["LGBM"][0].taus[kk]),
                  float(comps["LGBM"][0].taus[kk+1])])
else:
    nodes["lgbm_flat_example"] = None
m0 = np.load(OUTD / "U_LGBM.npy") == 0
nodes["lgbm_zero_rows_first_node_min"] = [float(q[si, m0, 1].min())
                                          for si in range(3)]  # Q(0.05)>0?
# BQN: 每种子支撑上端 Q(1)=c_D 逐行, 339 行 y vs 三种子上端
q1 = np.stack([c.Q_diag(np.ones(len(y))) for c in comps["BQN"]])  # (3, n)
np.save(OUTD / "bqn_Q1_per_seed.npy", q1)
u1 = np.load(OUTD / "U_BQN.npy") == 1
hi = q1.max(axis=0)
nodes["bqn_U1_rows"] = dict(
    n=int(u1.sum()),
    y_gt_all_seed_ends=int((y[u1] > hi[u1]).sum()),
    y_eq_some_seed_end=int((y[u1] == hi[u1]).sum()),
    y_lt_some_seed_end=int((y[u1] < hi[u1]).sum()),  # 应=0(混合F=1需全体到顶)
    min_gap=float((y[u1] - hi[u1]).min()),
    example_row=int(np.where(u1)[0][0]),
    example=dict(obs=float(y[u1][0]), q1_per_seed={str(SEEDS[si]):
                 float(q1[si][u1][0]) for si in range(3)}))
ev["nodes"] = nodes
report["evidence"] = ev
report["identity_pass"] = True
report["note"] = ("V=default_rng(42).random(n) 固定; U=clip(Fl+V*(Fr-Fl),0,1); "
                  "前9箱左闭右开, 末箱[0.9,1.0]含1; 描述性诊断, 无显著性带; "
                  "来源=确定性重建(与冻结q50/CRPS逐键零差), 命名待定")
(OUTD / "pit_report.json").write_text(json.dumps(report, indent=1))
pd.DataFrame({m: pit[m]["counts"] for m in pit},
             index=[f"[{i/10:.1f},{(i+1)/10:.1f})" if i < 9 else "[0.9,1.0]"
                    for i in range(10)]
             ).rename_axis("bin").to_csv(OUTD / "pit_counts.csv")
print("PIT 报告落盘", OUTD)
