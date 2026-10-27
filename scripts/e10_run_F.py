#!/usr/bin/env python
"""AWF E10 — F0-F 前向未见站(50 折) 三模型(预设结构, 三种子 CDF 混合)

运行号: AWF_E10_F_R1
依据: 统一执行方案 §四(S 设计)/§五(S/F 预设与混合)。逐折键来自冻结清单
outputs/bqn_lgbm_qrf/protocol/split_keys/split_keys_F.parquet。

预设: BQN 64/32·deg12·lr1e-3(内层站选轮数); QRF 500·minleaf50·mf1.0;
LGBM 300 树原空间默认(22 分位)。内层验证站 = 候选 ∩ 外层训练站。
正式预测 = 三种子等权混合; 逐行 CRPS/分位数/P(>15)/宽度带键保存。

输出: outputs/bqn_lgbm_qrf/F/{F_mixture.parquet, F_scores.json}
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/vhyang/zcode/awf/scripts")
from e10_models import (BQN, LGBM22, QRF, chunked_mixture_scores, encode,
                        verify_component)
from e10_protocol import station_hash20

DATA = Path("/mnt/d/data/awfdata")
OUT = Path("/home/vhyang/zcode/awf/outputs/bqn_lgbm_qrf/F")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [17, 42, 2026]
F0 = ["u10", "v10", "fcst_spd", "t2m", "prmsl", "rh2m", "lead_h",
      "month_sin", "month_cos", "lat", "lon", "elevation", "terrain"]
ONLY_MODEL = sys.argv[1] if len(sys.argv) > 1 else "all"   # bqn|qrf|lgbm|all


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
    return df.dropna(subset=["wspd"]).reset_index(drop=True)


def main() -> int:
    t00 = time.time()
    df = load()
    keys = pd.read_parquet("/home/vhyang/zcode/awf/outputs/bqn_lgbm_qrf/"
                           "protocol/split_keys/split_keys_F.parquet")
    stations = sorted(df.station.unique())
    inner_all = [s for s in stations if station_hash20(s)]
    # 原子候选(全数据口径, 仅检测用, 训练值重复>=50)
    vc = df.wspd.value_counts()
    train_atoms = vc[vc >= 50].index.values.astype(float)

    parts, fold_rows = [], []
    early = df.valid_time <= pd.Timestamp("2021-08-31 23:59")
    late = df.valid_time >= pd.Timestamp("2021-10-01 00:00")
    for i, s in enumerate(stations):
        test = (df.station == s) & late
        if int(test.sum()) == 0:
            continue
        outer_train = (df.station != s) & early
        inner_st = [c for c in inner_all if c != s]
        inner_mask = outer_train & df.station.isin(inner_st) \
            & (df.valid_time.dt.month == 8)
        fit_mask = outer_train & ~inner_mask
        tr, iv, te = df[fit_mask], df[inner_mask], df[test]
        row_evaluable = int(test.sum()) >= 50
        t0 = time.time()
        row = dict(fold=i, station=s, n_train=len(tr), n_inner=len(iv),
                   n_test=len(te), evaluable=bool(row_evaluable))
        preds = None
        if ONLY_MODEL in ("all", "bqn"):
            comps, bqn_fail = [], None
            for sd in SEEDS:
                try:
                    m = BQN(seed=sd, val_sub=1000, eval_every=2).fit(
                        encode(tr, F0, "bqn"), tr.wspd.values,
                        encode(iv, F0, "bqn"), iv.wspd.values)
                    comps.append(m.component(encode(te, F0, "bqn")))
                except Exception as e:                    # noqa
                    bqn_fail = f"seed{sd}:{type(e).__name__}:{e}"
                    break
            row["bqn_status"] = "ok" if bqn_fail is None else bqn_fail
            if bqn_fail is None:
                try:
                    preds_b = chunked_mixture_scores(comps, te.wspd.values,
                                                     chunk=1000,
                                                     train_values=train_atoms)
                except Exception as e:                # noqa
                    bqn_fail = f"score:{type(e).__name__}:{e}"
                    preds_b = pd.DataFrame(
                        {k: np.full(len(te), np.nan) for k in
                         ("crps", "q10", "q50", "q90", "p_gt15",
                          "width80", "_grid_check")})
                    row["bqn_status"] = bqn_fail
            else:
                preds_b = pd.DataFrame({k: np.full(len(te), np.nan) for k in
                                        ("crps", "q10", "q50", "q90",
                                         "p_gt15", "width80", "_grid_check")})
            row["bqn_crps"] = (round(float(preds_b.crps.mean()), 5)
                               if bqn_fail is None else None)
            row["bqn_mae"] = round(float(np.abs(preds_b.q50
                                                - te.wspd.values).mean()), 5)
            row["bqn_cov80"] = round(float(
                ((te.wspd.values >= preds_b.q10)
                 & (te.wspd.values <= preds_b.q90)).mean()), 4)
            preds = preds_b.add_prefix("bqn_")
        if ONLY_MODEL in ("all", "qrf"):
            comps = []
            for sd in SEEDS:
                m = QRF(seed=sd, n_jobs=24).fit(encode(tr, F0, "qrf"),
                                                tr.wspd.values)
                comps.append(m.component(encode(te, F0, "qrf")))
            preds_q = chunked_mixture_scores(comps, te.wspd.values,
                                             chunk=1000,
                                             train_values=train_atoms)
            row["qrf_crps"] = round(float(preds_q.crps.mean()), 5)
            row["qrf_mae"] = round(float(np.abs(preds_q.q50
                                                - te.wspd.values).mean()), 5)
            row["qrf_cov80"] = round(float(
                ((te.wspd.values >= preds_q.q10)
                 & (te.wspd.values <= preds_q.q90)).mean()), 4)
            preds = preds_q.add_prefix("qrf_") if preds is None else pd.concat(
                [preds, preds_q.add_prefix("qrf_")], axis=1)
        if ONLY_MODEL in ("all", "lgbm"):
            comps = []
            for sd in SEEDS:
                m = LGBM22(seed=sd, n_estimators=300).fit(tr[F0],
                                                          tr.wspd.values)
                comps.append(m.component(te[F0]))
            preds_l = chunked_mixture_scores(comps, te.wspd.values,
                                             chunk=1000,
                                             train_values=train_atoms)
            row["lgbm_crps"] = round(float(preds_l.crps.mean()), 5)
            row["lgbm_mae"] = round(float(np.abs(preds_l.q50
                                                 - te.wspd.values).mean()), 5)
            row["lgbm_cov80"] = round(float(
                ((te.wspd.values >= preds_l.q10)
                 & (te.wspd.values <= preds_l.q90)).mean()), 4)
            preds = preds_l.add_prefix("lgbm_") if preds is None else pd.concat(
                [preds, preds_l.add_prefix("lgbm_")], axis=1)
        base = te[["station", "init", "valid_time", "lead_h", "wspd",
                   "fcst_spd"]].rename(columns={"wspd": "obs",
                                                "fcst_spd": "gfs"}).copy()
        base["fold"] = i
        base = base.reset_index(drop=True)
        preds = preds.reset_index(drop=True)
        parts.append(pd.concat([base, preds], axis=1))
        row["elapsed_s"] = round(time.time() - t0, 1)
        fold_rows.append(row)
        print(f"[fold {i + 1}/50] {s}: {row.get('bqn_crps', '-')} "
              f"{row.get('qrf_crps', '-')} {row.get('lgbm_crps', '-')} "
              f"({row['elapsed_s']}s)", flush=True)
        fname = "F_mixture.parquet" if ONLY_MODEL == "all" else f"F_{ONLY_MODEL}.parquet"
        pd.concat(parts, ignore_index=True).to_parquet(OUT / fname,
                                                        index=False)  # 增量
    pd.DataFrame(fold_rows).to_csv(OUT / "F_folds.csv", index=False)
    fname = "F_mixture.parquet" if ONLY_MODEL == "all" else f"F_{ONLY_MODEL}.parquet"
    sm = pd.concat(parts, ignore_index=True)
    summary = {}
    evaluable = set(pd.read_parquet(
        "/home/vhyang/zcode/awf/outputs/bqn_lgbm_qrf/protocol/split_keys/"
        "split_keys_F.parquet").query("role=='test_outer'")
        .groupby("held_out_station").size().loc[lambda x: x >= 50].index)
    sm_eval = sm[sm.station.isin(evaluable)]
    for mdl in ("bqn", "qrf", "lgbm"):
        if f"{mdl}_crps" in sm:
            w = sm_eval[f"{mdl}_crps"].values
            mae = np.abs(sm_eval[f"{mdl}_q50"].values
                         - sm_eval.obs.values)
            mae = np.abs(sm[f"{mdl}_q50"].values - sm.obs.values)
            summary[mdl] = dict(
                pooled_crps=round(float(w.mean()), 4),
                pooled_mae=round(float(mae.mean()), 4),
                station_eq_crps=round(float(
                    sm_eval.groupby("station")[f"{mdl}_crps"]
                    .mean().mean()), 4),
                cov80=round(float(((sm_eval.obs >= sm_eval[f"{mdl}_q10"])
                                   & (sm_eval.obs <= sm_eval[f"{mdl}_q90"]))
                                  .mean()), 4),
                gfs_mae=round(float(np.abs(sm_eval.gfs.values
                                           - sm_eval.obs.values).mean()), 4),
                n_rows=len(sm_eval), n_all_rows=len(sm))
    (OUT / "F_scores.json").write_text(json.dumps(dict(
        run_id="AWF_E10_F_R1", utc=datetime.now(timezone.utc).isoformat(),
        model=ONLY_MODEL, summary=summary,
        preset=dict(BQN="64/32 deg12 lr1e-3 inner-epoch-select (val_sub=1000, eval_every=2, S/F 提速, 已登记)",
                    QRF="500 minleaf50 mf1.0", LGBM="300 trees spatial "
                    "defaults 22q"),
        seeds=SEEDS, elapsed=round(time.time() - t00, 1)), indent=2))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
