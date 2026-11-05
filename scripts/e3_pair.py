#!/usr/bin/env python
"""Pair GFS forecasts with station observations at matching valid times.

Scalar wind speed is retained independently of direction availability.
Direction statistics use observed speed >= 2.0 m/s and a recorded direction;
u/v values are missing below this speed or when direction is missing.
err_spd is observation minus forecast; later bias summaries use the opposite
sign. A station-valid-time observation may occur at several lead times.
Run ID: AWF_E3_PAIR_R1."""
import os
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("AWF_DATA_ROOT", "data"))
RAW3H = DATA / "raw" / "antaws" / "3_hourly" / "3_hourly"
STATION_FILE = DATA / "derived" / "station_qc" / "usable_stations.csv"
GFS_DIR = DATA / "derived" / "paired" / "gfs_point"
OUT = DATA / "derived" / "paired"

YEAR = 2021
MIN_WS_FOR_DIR = 2.0   # 风速<2 m/s 时风向不可信, u/v 置 NaN (速度仍保留)


def load_obs(station: str) -> pd.DataFrame | None:
    f = RAW3H / f"{station}_3h.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, header=None, skiprows=1,
                     names=["year", "month", "day", "hour", "t2m", "psfc",
                            "wspd", "wdir", "rh"],
                     na_values=["NA", ""], engine="c", encoding="latin1")
    df = df[df.year == YEAR]
    if df.empty:
        return None
    ts = pd.to_datetime(dict(year=df.year, month=df.month, day=df.day,
                             hour=df.hour), errors="coerce")
    df = df.assign(valid_time=ts).dropna(subset=["valid_time"])
    return df


def main() -> int:
    run_id = "AWF_E3_PAIR_R1"
    st = pd.read_csv(STATION_FILE)
    print(f"[{run_id}] stations={len(st)}")

    # --- 观测端 ---
    obs_parts = []
    for s in st.station:
        df = load_obs(s)
        if df is not None:
            obs_parts.append(df.assign(station=s))
    obs = pd.concat(obs_parts, ignore_index=True)
    # 风向 -> u/v (圆周转换, 低风置 NaN)
    rad = np.deg2rad(obs.wdir.values)
    spd = obs.wspd.values
    trust = spd >= MIN_WS_FOR_DIR
    obs["obs_u"] = np.where(trust, -spd * np.sin(rad), np.nan)
    obs["obs_v"] = np.where(trust, -spd * np.cos(rad), np.nan)
    obs["dir_trust"] = trust
    obs = obs[["station", "valid_time", "wspd", "wdir", "obs_u", "obs_v",
               "dir_trust", "t2m", "psfc", "rh"]]
    print(f"[{run_id}] obs rows ({YEAR})={len(obs)}")

    # --- 预报端 (增量合并已下载的每日 parquet; 冒烟测试在个别文件留了重复行, 先去重) ---
    gfs_parts = []
    for f in sorted(GFS_DIR.glob("gfs_*.parquet")):
        d = pd.read_parquet(f)
        n0 = len(d)
        d = d.drop_duplicates(subset=["station", "lead_h", "valid_time"], keep="first")
        gfs_parts.append(d)
    if not gfs_parts:
        print("no gfs parquet yet"); return 1
    gfs = pd.concat(gfs_parts, ignore_index=True)
    gfs["valid_time"] = pd.to_datetime(gfs["valid_time"])
    n_dup_removed = sum(len(x) for x in gfs_parts) and \
        int(sum(len(pd.read_parquet(f)) for f in GFS_DIR.glob('gfs_*.parquet')) - len(gfs))
    print(f"[{run_id}] gfs rows={len(gfs)} (dup_removed={n_dup_removed}) days={gfs.init.nunique()}")

    # --- 配对 ---
    paired = pd.merge(
        obs, gfs,
        on=["station", "valid_time"], how="inner",
        validate="one_to_many",   # 同一 valid_time 可对应多个 lead
    )
    paired["err_u"] = paired.obs_u - paired.u10
    paired["err_v"] = paired.obs_v - paired.v10
    paired["err_spd"] = paired.wspd - np.hypot(paired.u10, paired.v10)

    paired.to_parquet(OUT / f"paired_{YEAR}.parquet", index=False)

    # --- 自检 ---
    by_lead = paired.groupby("lead_h").agg(
        n=("station", "size"),
        err_spd_mean=("err_spd", "mean"),
        err_spd_std=("err_spd", "std"),
        abs_err_spd=("err_spd", lambda x: float(np.abs(x).mean())),
    ).round(3)
    selfcheck = dict(
        run_id=run_id,
        utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        n_obs=len(obs), n_gfs=len(gfs), n_paired=len(paired),
        n_stations_paired=int(paired.station.nunique()),
        by_lead=by_lead.reset_index().to_dict("records"),
    )
    (OUT / "e3_pair_run.json").write_text(
        json.dumps(selfcheck, ensure_ascii=False, indent=2, default=str))
    print(by_lead)
    return 0


if __name__ == "__main__":
    sys.exit(main())
