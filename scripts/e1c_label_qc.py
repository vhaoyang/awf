#!/usr/bin/env python
"""AWF E1C — 观测风速标签质量预检（大风传感器失效风险, 风险清单#3）

运行号: AWF_E1C_LABELQC_R1 (服务器端自编, 非本侧任务号)

检查项 (每站 2021 年 3-hourly wspd 序列):
  F  frozen   连续 >=6 步(18h)完全相同的非零值 — 冻住/卡死风速计
  Z  zero-run 连续 >=12 步(36h)精确 0.0 — 疑似死传感器(内陆静稳真实存在,
              仅登记不作废)
  S  satured  wspd >= 45 m/s — 接近仪器量程上限, 人工复核清单
  P  spike    相邻 3h |Δwspd| >= 15 m/s — katabatic 阵风可真, 计数观察

输出:
  derived/station_qc/obs_label_qc_2021.csv  逐站登记
  derived/station_qc/obs_label_qc_2021.json 汇总(占比口径: 站点-小时)
说明: 只登记不删数据; E4 敏感性分析可用 flag 列重算判据。
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/mnt/d/data/awfdata")
RAW3H = DATA / "raw" / "antaws" / "3_hourly" / "3_hourly"
STATION_FILE = DATA / "derived" / "station_qc" / "usable_stations.csv"
YEAR = 2021
FLAT_MIN_RUN, FLAT_MIN_VAL = 6, 0.5     # >=18h 相同且 >0.5 m/s
ZERO_MIN_RUN = 12                        # >=36h 精确 0
SAT_THR = 45.0
SPIKE_THR = 15.0


def runs_of_const(x: np.ndarray, min_run: int, lo: float | None, hi: float | None):
    """满足值域条件的连续相同值 run 总数/最长/总步数。"""
    n_run = max_run = tot = 0
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[j + 1] == x[i]:
            j += 1
        L = j - i + 1
        v = x[i]
        ok = (L >= min_run and not np.isnan(v)
              and (lo is None or v >= lo) and (hi is None or v <= hi))
        if ok:
            n_run += 1
            tot += L
            max_run = max(max_run, L)
        i = j + 1
    return n_run, max_run, tot


def main() -> int:
    st = pd.read_csv(STATION_FILE)
    rows = []
    for s in st.station:
        f = RAW3H / f"{s}_3h.csv"
        if not f.exists():
            rows.append(dict(station=s, n_obs=0)); continue
        df = pd.read_csv(f, header=None, skiprows=1,
                         names=["y", "m", "d", "h", "t2m", "psfc", "wspd", "wdir", "rh"],
                         na_values=["NA", ""], engine="c", encoding="latin1")
        x = df.loc[df.y == YEAR, "wspd"].to_numpy(float)
        n = len(x)
        if n == 0:
            rows.append(dict(station=s, n_obs=0)); continue
        fr = runs_of_const(x, FLAT_MIN_RUN, FLAT_MIN_VAL, None)
        zr = runs_of_const(x, ZERO_MIN_RUN, None, 0.0 + 1e-9)
        sat = int((x >= SAT_THR).sum())
        spike = int((np.abs(np.diff(x)) >= SPIKE_THR).sum())
        rows.append(dict(
            station=s, n_obs=n,
            frozen_runs=fr[0], frozen_max_h=fr[1] * 3, frozen_hours=fr[2] * 3,
            zerorun_n=zr[0], zerorun_max_h=zr[1] * 3, zerorun_hours=zr[2] * 3,
            n_saturated=sat, n_spike=spike,
            pct_flagged=round(100.0 * (fr[2] * 3 + zr[2] * 3) / max(n, 1), 2),
        ))
    out = pd.DataFrame(rows).sort_values("pct_flagged", ascending=False)
    out.to_csv(DATA / "derived" / "station_qc" / "obs_label_qc_2021.csv", index=False)

    tot_obs = int(out.n_obs.sum())
    tot_flag_h = int(out.frozen_hours.sum() + out.zerorun_hours.sum())
    summary = dict(
        run_id="AWF_E1C_LABELQC_R1",
        utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        n_stations=len(out), total_obs_station_hours=tot_obs,
        flagged_station_hours=tot_flag_h,
        pct_flagged_total=round(100.0 * tot_flag_h / max(tot_obs, 1), 3),
        n_stations_with_frozen=int((out.frozen_runs.fillna(0) > 0).sum()),
        n_stations_with_longzero=int((out.zerorun_n.fillna(0) > 0).sum()),
        n_saturated_total=int(out.n_saturated.fillna(0).sum()),
        n_spike_total=int(out.n_spike.fillna(0).sum()),
        thresholds=dict(frozen=f">={FLAT_MIN_RUN}步相同非零(>=0.5m/s), 即>={FLAT_MIN_RUN*3}h",
                        zero=f">={ZERO_MIN_RUN}步精确0, 即>={ZERO_MIN_RUN*3}h",
                        saturated=f">={SAT_THR} m/s", spike=f"|Δ|>={SPIKE_THR} m/s/3h"),
        note="只登记不删数据; 口径分母=站点-小时(74站 2021 全年 3-hourly)",
        worst10=out.head(10).to_dict("records"),
    )
    (DATA / "derived" / "station_qc" / "obs_label_qc_2021.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "worst10"},
                     ensure_ascii=False, indent=2))
    print(out.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
