#!/usr/bin/env python
"""AWF E10 — F0-T 主比较：三模型 8 候选 × 3 种子搜索 + 正式混合预测

运行号: AWF_E10_T_R1
依据: 统一执行方案 §四(T 设计)/§五(候选与种子)/§六(评分); 评分器为
12-R1 修复后 e10_core v4(未用于本次之前的任何选择)。

流程:
  1) 每模型 8 预登记候选 × 种子 {17,42,2026}: 训练=3–8月, 内层=9月;
     选择准则=9月完整分布 CRPS(单种子分布, 逐种子记录后按种子均值选)。
  2) 选中配置下三种子模型 -> 10–12月测试(18,235 行)正式预测 =
     等权 CDF 混合; 输出逐行 CRPS/分位数(广义逆)/P(>15)/宽度。
  3) 配对差值日期块 bootstrap(7d 主, 1/3d 敏感, 2000 次): BQN−LGBM、
     BQN−QRF、QRF−LGBM、LGBM−EMOS(EMOS 复用冻结预测, 同键)。
  4) 全程记录候选/失败/时间; 预测带键保存。

输出: outputs/bqn_lgbm_qrf/T/{search_results.json, T_mixture.parquet,
  T_scores.json, T_bootstrap.csv}
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/vhyang/zcode/awf/scripts")
from e10_core import mixture_crps, mixture_stats
from e10_models import (BQN, LGBM22, QRF, chunked_mixture_scores,
                        encode, single_crps, slice_component, verify_component)
from e8_core import block_bootstrap

DATA = Path("/mnt/d/data/awfdata")
OUT = Path("/home/vhyang/zcode/awf/outputs/bqn_lgbm_qrf/T")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [17, 42, 2026]
THR = 15.0

CAND_BQN = [dict(degree=d, width=w, lr=lr)
            for d in (8, 12) for w in ((32, 16), (64, 32))
            for lr in (1e-3, 3e-4)]
CAND_QRF = [dict(min_samples_leaf=m, max_features=f)
            for m in (10, 30, 50, 100) for f in (0.7, 1.0)]
CAND_LGBM = [dict(num_leaves=l, min_data_in_leaf=m, learning_rate=r)
             for l in (15, 31) for m in (50, 100) for r in (0.03, 0.05)]


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
    return df


def main() -> int:
    t_start = time.time()
    df = load()
    F0 = ["u10", "v10", "fcst_spd", "t2m", "prmsl", "rh2m", "lead_h",
          "month_sin", "month_cos", "lat", "lon", "elevation", "terrain"]
    tr = df[df.month <= 8].reset_index(drop=True)
    va = df[df.month == 9].reset_index(drop=True)
    te = df[df.month >= 10].reset_index(drop=True)
    yv = va.wspd.values
    print(f"[data] train {len(tr)} val {len(va)} test {len(te)}", flush=True)

    search = {"BQN": [], "QRF": [], "LGBM": []}
    best = {}
    reuse = None
    if (OUT / "search_results.json").exists():
        reuse = json.loads((OUT / "search_results.json").read_text())
        search = reuse["search"]
        print("[reuse] 已有搜索结果, 跳过训练搜索", flush=True)

    # ---------------- BQN 搜索(GPU) ----------------
    Xtr, Xva = encode(tr, F0, "bqn"), encode(va, F0, "bqn")
    import os
    for ci, cfg in enumerate([] if reuse else CAND_BQN):
        seed_scores = []
        for sd in SEEDS:
            t0 = time.time()
            try:
                m = BQN(seed=sd, **cfg).fit(Xtr, tr.wspd.values, Xva, yv)
                comp = m.component(Xva)
                crps = float(np.mean(single_crps(comp, yv, chunk=2500)))
                search["BQN"].append(dict(
                    cand=ci, cfg={k: str(v) for k, v in cfg.items()}, seed=sd,
                    sep_crps=round(crps, 6), best_val_crps=round(
                        float(m.best_val_crps), 6),
                    elapsed=round(time.time() - t0, 1), status="ok"))
            except Exception as e:                                # noqa
                search["BQN"].append(dict(cand=ci, cfg=str(cfg), seed=sd,
                                          elapsed=round(time.time() - t0, 1),
                                          status=f"fail:{e}"))
                crps = np.inf
            seed_scores.append(crps)
        print(f"[BQN] cand{ci} {cfg} mean_sep_crps="
              f"{np.mean(seed_scores):.4f}", flush=True)
        best.setdefault("BQN", []).append((float(np.mean(seed_scores)), ci))
    if reuse:
        sel = reuse["selected"]["BQN"]
        best_cand_BQN = dict(degree=int(sel["degree"]),
                             width=eval(sel["width"]), lr=float(sel["lr"]))
    else:
        best_cand_BQN = CAND_BQN[min(best["BQN"])[1]]

    # ---------------- QRF 搜索(n_jobs=24) ----------------
    Xtrq, Xvaq = encode(tr, F0, "qrf"), encode(va, F0, "qrf")
    for ci, cfg in enumerate([] if reuse else CAND_QRF):
        seed_scores = []
        for sd in SEEDS:
            t0 = time.time()
            m = QRF(seed=sd, n_jobs=24, **cfg).fit(Xtrq, tr.wspd.values)
            comp = m.component(Xvaq)
            crps = float(np.mean(single_crps(comp, yv, chunk=2500)))
            search["QRF"].append(dict(cand=ci, cfg=cfg, seed=sd,
                                      sep_crps=round(crps, 6),
                                      elapsed=round(time.time() - t0, 1),
                                      status="ok"))
            seed_scores.append(crps)
        print(f"[QRF] cand{ci} {cfg} mean={np.mean(seed_scores):.4f}",
              flush=True)
        best.setdefault("QRF", []).append((float(np.mean(seed_scores)), ci))
    if reuse:
        best_cand_QRF = reuse["selected"]["QRF"]
    else:
        best_cand_QRF = CAND_QRF[min(best["QRF"])[1]]

    # ---------------- LGBM 搜索(6线程×4进程并行经 ProcessPool) ----------------
    if not reuse:
        import multiprocessing as mp
        with mp.Pool(4) as pool:
            jobs = [(ci, cfg, sd) for ci, cfg in enumerate(CAND_LGBM)
                    for sd in SEEDS]
            res = pool.starmap(_fit_lgbm_one,
                               [(ci, cfg, sd, "T") for ci, cfg, sd in jobs])
        for r in res:
            search["LGBM"].append(r)
        agg = (pd.DataFrame([r for r in res if r["status"] == "ok"])
               .groupby("cand")["sep_crps"].mean())
        best_cand_LGBM = CAND_LGBM[int(agg.idxmin())]
    else:
        best_cand_LGBM = reuse["selected"]["LGBM"]

    (OUT / "search_results.json").write_text(json.dumps(dict(
        search=search, selected=dict(BQN={k: str(v) for k, v in
                                          best_cand_BQN.items()},
                                     QRF=best_cand_QRF,
                                     LGBM=best_cand_LGBM)), indent=2,
        default=str))
    print("[selected]", best_cand_BQN, best_cand_QRF, best_cand_LGBM,
          flush=True)

    # ---------------- 正式预测: 三种子组件 -> 测试集混合 ----------------
    Xte, Xteq = encode(te, F0, "bqn"), encode(te, F0, "qrf")
    comps = {"BQN": [], "QRF": [], "LGBM": []}
    for sd in SEEDS:
        comps["BQN"].append(BQN(seed=sd, **best_cand_BQN).fit(
            Xtr, tr.wspd.values, Xva, yv).component(Xte))
        comps["QRF"].append(QRF(seed=sd, n_jobs=24, **best_cand_QRF).fit(
            Xtrq, tr.wspd.values).component(Xteq))
        comps["LGBM"].append(_fit_lgbm_final(best_cand_LGBM, sd, tr, te, F0))
    yte = te.wspd.values
    # 原子候选: 训练 y 中重复>=50 的取值(QRF 原子检测)
    vc = tr.wspd.value_counts()
    train_atoms = vc[vc >= 50].index.values.astype(float)
    scores, preds = {}, {}
    for model, cc in comps.items():
        part = OUT / f"T_partial_{model}.parquet"
        if part.exists():
            print(f"[cache] {model} 复用部分结果", flush=True)
            preds[model] = pd.read_parquet(part)
            row = preds[model]
            elapsed = 0.0
        else:
            t0 = time.time()
            row = chunked_mixture_scores(cc, yte, chunk=1500,
                                         train_values=train_atoms)
            row.to_parquet(part, index=False)
            preds[model] = row
            elapsed = time.time() - t0
        scores[model] = dict(
            crps=round(float(row.crps.mean()), 4),
            mae=round(float(np.abs(row.q50 - yte).mean()), 4),
            bias=round(float((row.q50 - yte).mean()), 4),
            cov80=round(float(((yte >= row.q10) & (yte <= row.q90)).mean()), 4),
            width80=round(float(row.width80.mean()), 4),
            brier=round(float(((row.p_gt15 - (yte > THR)) ** 2).mean()), 6),
            verify=[verify_component(slice_component(c, slice(0, 50)))
                    for c in cc],
            elapsed=round(elapsed, 1))
        print(f"[score] {model}: {scores[model]['crps']} "
              f"({scores[model]['elapsed']}s)", flush=True)
    # 逐行预测落盘(带键)
    out = te[["station", "init", "valid_time", "lead_h", "wspd",
              "fcst_spd"]].rename(columns={"wspd": "obs",
                                           "fcst_spd": "gfs"}).copy()
    for model in preds:
        for k in ("crps", "q10", "q50", "q90", "p_gt15", "width80"):
            out[f"{model}_{k}"] = preds[model][k].values
    out.to_parquet(OUT / "T_mixture.parquet", index=False)

    # ---------------- EMOS 参照(冻结预测同键合并) ----------------
    emos = pd.read_parquet("/home/vhyang/zcode/awf/outputs/review_v3/"
                           "predictions_v3.parquet")
    key = ["station", "init", "valid_time", "lead_h"]
    m = out[key].merge(emos[key + ["emos_mu", "emos_sd"]], on=key,
                       how="left", validate="one_to_one")
    from e10_core import GaussComponent
    emos_crps = np.empty(len(out))
    for s in range(0, len(out), 2500):
        e = min(s + 2500, len(out))
        g = GaussComponent(0, 1)          # 占位, 用向量化闭式重算
        from e8_core import crps_normal_closed
        emos_crps[s:e] = crps_normal_closed(
            yte[s:e], m.emos_mu.values[s:e], m.emos_sd.values[s:e])
    scores["EMOS_ref"] = dict(crps=round(float(emos_crps.mean()), 4))

    # ---------------- 配对差值日期块 bootstrap ----------------
    dates = pd.to_datetime(out.init).dt.date.values
    boot_rows = []
    pairs = [("BQN", "LGBM"), ("BQN", "QRF"), ("QRF", "LGBM"),
             ("LGBM", "EMOS")]
    row_crps = {k: preds[k]["crps"].values for k in preds}
    row_crps["EMOS"] = emos_crps
    for a, b in pairs:
        dvec = row_crps[a] - row_crps[b]
        for bl in (7, 3, 1):
            st = block_bootstrap(dvec, dates, bl, n_boot=2000, seed=42)
            lo, hi = np.percentile(st, [2.5, 97.5])
            boot_rows.append(dict(pair=f"{a}-{b}", block_days=bl,
                                  mean_diff=round(float(dvec.mean()), 6),
                                  ci_lo=round(float(lo), 6),
                                  ci_hi=round(float(hi), 6),
                                  excl0=bool(lo > 0 or hi < 0)))
    pd.DataFrame(boot_rows).to_csv(OUT / "T_bootstrap.csv", index=False)
    (OUT / "T_scores.json").write_text(json.dumps(dict(
        run_id="AWF_E10_T_R1", utc=datetime.now(timezone.utc).isoformat(),
        selected=dict(BQN={k: str(v) for k, v in best_cand_BQN.items()},
                      QRF=best_cand_QRF, LGBM=best_cand_LGBM),
        scores=scores, seeds=SEEDS, thr=THR,
        train_atoms=list(map(float, train_atoms)),
        elapsed_total=round(time.time() - t_start, 1)), indent=2,
        default=str))
    print(json.dumps(scores, indent=1, default=str))
    print("[boot]", pd.DataFrame(boot_rows).to_string(index=False))
    return 0


def _fit_lgbm_one(ci, cfg, sd, tag):
    """子进程: 单候选单种子 LGBM(6 线程), 9 月早停, 返回 9 月完整 CRPS。"""
    import numpy as np
    import pandas as pd
    import time
    df = load()
    F0 = ["u10", "v10", "fcst_spd", "t2m", "prmsl", "rh2m", "lead_h",
          "month_sin", "month_cos", "lat", "lon", "elevation", "terrain"]
    tr = df[df.month <= 8].reset_index(drop=True)
    va = df[df.month == 9].reset_index(drop=True)
    from e10_models import LGBM22, single_crps
    t0 = time.time()
    try:
        m = LGBM22(seed=sd, n_estimators=1000,
                   early_stopping=(va[F0], va.wspd.values), **cfg).fit(
            tr[F0], tr.wspd.values)
        comp = m.component(va[F0])
        crps = float(np.mean(single_crps(comp, va.wspd.values[:6000],
                                         chunk=2000)))
        return dict(cand=ci, cfg=cfg, seed=sd, sep_crps=round(crps, 6),
                    elapsed=round(time.time() - t0, 1), status="ok",
                    best_iters=m.best_iters[:3])
    except Exception as e:                                        # noqa
        return dict(cand=ci, cfg=cfg, seed=sd,
                    elapsed=round(time.time() - t0, 1),
                    status=f"fail:{e}")


def _fit_lgbm_final(cfg, sd, tr, te, F0):
    """正式 LGBM: 训练=Mar–Aug, 9月早停(与搜索同协议); 返回测试组件。"""
    from e10_models import LGBM22
    import pandas as pd
    df = load()
    sep = df[df.month == 9]
    m = LGBM22(seed=sd, n_estimators=1000,
               early_stopping=(sep[F0], sep.wspd.values), **cfg).fit(
        tr[F0], tr.wspd.values)
    return m.component(te[F0])


if __name__ == "__main__":
    sys.exit(main())
