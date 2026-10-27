#!/usr/bin/env python
"""AWF E10 — S/F 事后汇总：三模型配对合并、池化/站等权、逐站、失败折统计

运行号: AWF_E10_SUM_R1
输入: outputs/bqn_lgbm_qrf/{S,F}/{S,F}_{bqn,qrf,lgbm}.parquet(逐行带键)
输出: outputs/bqn_lgbm_qrf/{S,F}/summary_v2.json + per-station CSV
      (运行器内嵌汇总有 bug, 本脚本为权威汇总)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/vhyang/zcode/awf/scripts")
from e8_core import block_bootstrap

BASE = Path("/home/vhyang/zcode/awf/outputs/bqn_lgbm_qrf")
KEYS = ["station", "init", "valid_time", "lead_h"]
MODELS = ["bqn", "qrf", "lgbm"]


def summarize(design: str, boot=True):
    d = BASE / design
    frames = {}
    for m in MODELS:
        f = d / f"{design}_{m}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        # S1/F1(e10_run_F1) 无前缀评分列 -> 归一化为 {model}_*
        if "crps" in df.columns:
            df = df.rename(columns={
                k: f"{m}_{k}" for k in ("crps", "q10", "q50", "q90",
                                        "p_gt15", "width80")})
        # 同键多折重复(断点续跑) -> 去重取首
        df = df.drop_duplicates(subset=KEYS + ["fold"], keep="first")
        frames[m] = df
    # 模型间按键对齐(qrf 为基准全量; 其他模型可能缺折/失败行)
    base = frames["qrf"][KEYS + ["obs", "gfs", "fold"]]
    merged = base.copy()
    for m, df in frames.items():
        cols = KEYS + [f"{m}_{k}" for k in ("crps", "q10", "q50", "q90",
                                            "p_gt15", "width80")]
        have = [c for c in cols if c in df.columns]
        merged = merged.merge(df[have].drop_duplicates(subset=KEYS,
                                                       keep="first"),
                              on=KEYS, how="left")
    y = merged.obs.values
    out = {"design": design, "n_rows_base": len(merged)}
    for m in frames:
        c = f"{m}_crps"
        ok = merged[c].notna().values if c in merged else np.zeros(len(merged),
                                                                  bool)
        st = dict(
            n_scored=int(ok.sum()),
            n_fit_or_score_failed=int((~ok).sum()),
            folds_scored=int(merged.loc[ok, "fold"].nunique()) if ok.any() else 0,
            pooled_crps=round(float(merged.loc[ok, c].mean()), 4) if ok.any() else None,
            pooled_mae=round(float(np.abs(merged.loc[ok, f"{m}_q50"] - y[ok]).mean()), 4) if ok.any() else None,
            pooled_bias=round(float((merged.loc[ok, f"{m}_q50"] - y[ok]).mean()), 4) if ok.any() else None,
            station_eq_crps=round(float(merged.loc[ok].groupby("station")[c].mean().mean()), 4) if ok.any() else None,
            cov80=round(float(((y[ok] >= merged.loc[ok, f"{m}_q10"].values)
                               & (y[ok] <= merged.loc[ok, f"{m}_q90"].values)).mean()), 4) if ok.any() else None,
            mean_width80=round(float(merged.loc[ok, f"{m}_width80"].mean()), 4) if ok.any() else None)
        # F 设计: 另算 42 可评站子集(>=50 测试行)
        if design == "F":
            cnt = base.groupby("station").size()
            evaluable = cnt[cnt >= 50].index
            sub_ok = ok & merged.station.isin(evaluable).values
            st["evaluable42_crps"] = round(
                float(merged.loc[sub_ok, c].mean()), 4) if sub_ok.any() else None
            st["evaluable42_mae"] = round(
                float(np.abs(merged.loc[sub_ok, f"{m}_q50"]
                             - y[sub_ok]).mean()), 4) if sub_ok.any() else None
            st["evaluable_stations"] = int(sub_ok.sum() and
                                           merged.loc[sub_ok].station.nunique())
        out[m] = st
    out["gfs"] = dict(pooled_mae=round(float(np.abs(merged.gfs - y).mean()), 4),
                      pooled_bias=round(float((merged.gfs - y).mean()), 4))
    # 逐站表
    per = merged.groupby("station").apply(lambda g: pd.Series({
        **{f"{m}_crps": (round(float(g[f"{m}_crps"].mean()), 4)
                         if f"{m}_crps" in g and g[f"{m}_crps"].notna().any()
                         else None) for m in frames},
        "gfs_mae": round(float(np.abs(g.gfs - g.obs).mean()), 4),
        "n": len(g)}), include_groups=False).reset_index()
    per.to_csv(d / f"{design}_per_station.csv", index=False)
    # 配对差值日期块 bootstrap(共同成功行)
    if boot:
        dates = pd.to_datetime(merged.init).dt.date.values
        rows = []
        pairs = [(a, b) for i, a in enumerate(MODELS if "bqn" in frames
                                              else ["qrf", "lgbm"])
                 for b in MODELS if a < b]
        for a, b in pairs:
            ca, cb = f"{a}_crps", f"{b}_crps"
            if ca not in merged or cb not in merged:
                continue
            ok = merged[ca].notna() & merged[cb].notna()
            if not ok.any():
                continue
            dvec = (merged.loc[ok, ca] - merged.loc[ok, cb]).values
            dts = dates[ok]
            for bl in (7, 3, 1):
                stb = block_bootstrap(dvec, dts, bl, n_boot=2000, seed=42)
                lo, hi = np.percentile(stb, [2.5, 97.5])
                rows.append(dict(pair=f"{a}-{b}", block_days=bl,
                                 mean_diff=round(float(dvec.mean()), 6),
                                 ci_lo=round(float(lo), 6),
                                 ci_hi=round(float(hi), 6),
                                 excl0=bool(lo > 0 or hi < 0),
                                 n_common=int(ok.sum())))
        pd.DataFrame(rows).to_csv(d / f"{design}_bootstrap.csv", index=False)
        out["bootstrap"] = rows
    (d / "summary_v2.json").write_text(json.dumps(out, ensure_ascii=False,
                                                  indent=2))
    print(f"[{design}]", json.dumps({m: out.get(m, {}) for m in frames},
                                    ensure_ascii=False, indent=1))
    return out


if __name__ == "__main__":
    for design in sys.argv[1:] or ["S", "F"]:
        summarize(design)
