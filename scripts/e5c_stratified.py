#!/usr/bin/env python
"""AWF E5C — E5 测试集分层评估（lead × 地形 × 月份; 图数据已含预测落盘）

运行号: AWF_E5C_STRAT_R1 (服务器端自编, 非本侧任务号)

输入: outputs/e5_baseline/test_predictions.parquet (E5 时间块划分的测试预测)
输出: outputs/e5_baseline/stratified/{by_lead.csv, by_terrain_lead.csv,
       by_month.csv, by_station.csv, strat_summary.json}
口径: 分母=各层内站点-小时行; CRPS3q 仅 LGBM(其余无分布)。
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/mnt/d/data/awfdata")
PRED = Path("/home/vhyang/zcode/awf/outputs/e5_baseline/test_predictions.parquet")
REG = DATA / "derived" / "station_qc" / "station_sensor_registry.csv"
OUT = Path("/home/vhyang/zcode/awf/outputs/e5_baseline/stratified")
TAIL = 15.0
QS = [10, 50, 90]


def metrics(g: pd.DataFrame) -> dict:
    y = g.wspd.values
    d = dict(
        n=len(g),
        gfs_mae=float(np.abs(g.gfs_spd.values - y).mean()),
        lgbm_mae=float(np.abs(g.lgbm_spd_q50.values - y).mean()),
        gfs_bias=float((g.gfs_spd.values - y).mean()),
        lgbm_bias=float((g.lgbm_spd_q50.values - y).mean()),
    )
    pin = []
    for q in QS:
        z = y - g[f"lgbm_spd_q{q}"].values
        tau = q / 100
        pin.append(float(np.mean(np.maximum(tau * z, (tau - 1) * z))))
    d["lgbm_crps3q"] = float(2 * np.mean(pin))
    t = y > TAIL
    if t.sum() >= 10:
        d["tail_n"] = int(t.sum())
        d["gfs_tail_bias"] = float((g.gfs_spd.values[t] - y[t]).mean())
        d["lgbm_tail_bias"] = float((g.lgbm_spd_q50.values[t] - y[t]).mean())
    return d


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = "AWF_E5C_STRAT_R1"
    df = pd.read_parquet(PRED)
    reg = pd.read_csv(REG)
    df = df.merge(reg[["station", "terrain", "institution"]], on="station",
                  how="left")
    df["month"] = df.valid_time.dt.month

    by_lead = pd.DataFrame(
        [dict(lead_h=lh, **metrics(g)) for lh, g in df.groupby("lead_h")])
    by_tl = pd.DataFrame(
        [dict(terrain=te, lead_h=lh, **metrics(g))
         for (te, lh), g in df.groupby(["terrain", "lead_h"])])
    by_month = pd.DataFrame(
        [dict(month=m, **metrics(g)) for m, g in df.groupby("month")])
    by_station = pd.DataFrame(
        [dict(station=s, terrain=str(g.terrain.iloc[0]), **metrics(g))
         for s, g in df.groupby("station")]).sort_values("lgbm_mae")

    for name, t in [("by_lead", by_lead), ("by_terrain_lead", by_tl),
                    ("by_month", by_month), ("by_station", by_station)]:
        t.round(3).to_csv(OUT / f"{name}.csv", index=False)

    summary = dict(
        run_id=run_id, utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        n_rows=len(df),
        denominator="层内站点-小时行; CRPS3q 仅 LGBM",
        lgbm_wins_by_lead={
            int(r.lead_h): round(float(r.lgbm_mae < r.gfs_mae), 3)
            for _, r in by_lead.iterrows()},
        lead_rows=by_lead.round(3).to_dict("records"),
    )
    (OUT / "strat_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(by_lead.round(3).to_string(index=False))
    print(by_month.round(3).to_string(index=False))
    print("->", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
