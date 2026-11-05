#!/usr/bin/env python
"""Screen AntAWS station records and summarize usable observations.

The initial station screen uses the configured overlap years; subsequent GFS
availability and pairing determine the study sample. This script does not
adjust observed wind speeds to a common sensor height.
Run ID: AWF_E1_QC_R1."""
import os
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile

DATA = Path(os.environ.get("AWF_DATA_ROOT", "data"))
RAW3H = DATA / "raw" / "antaws" / "3_hourly" / "3_hourly"
SHP = DATA / "raw" / "antaws" / "shapefiles" / "Shp" / "267AWS.shp"
OUT = DATA / "derived" / "station_qc"

# 筛选判据 (S1)
MIN_WIND_INTEGRITY = 0.50     # 3-hourly 风速观测完整性下限
GFS_MIN_YEAR, GFS_MAX_YEAR = 2015, 2021   # GFS 0.25° 配对窗口 (AntAWS 止于 2021-12-31)
MIN_OVERLAP_YEARS = 3

COLS = ["year", "month", "day", "hour", "t2m", "psfc", "wspd", "wdir", "rh"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_station_csv(path: Path) -> pd.DataFrame:
    # 表头为 GBK 乱码, 直接跳过整行, 按列位置命名:
    # 0=Year 1=Month 2=Day 3=Hour(UTC) 4=T 5=P 6=WS 7=WD 8=RH
    df = pd.read_csv(
        path, header=None, skiprows=1, names=COLS,
        na_values=["NA", ""], engine="c", encoding="latin1",
    )
    return df


def read_station_meta() -> pd.DataFrame:
    r = shapefile.Reader(str(SHP))
    fields = [f[0] for f in r.fields[1:]]
    recs = [dict(zip(fields, rec)) for rec in r.records()]
    return pd.DataFrame(recs).rename(columns={"zhandian": "station"})


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = "AWF_E1_QC_R1"
    t0 = datetime.now(timezone.utc)

    meta = read_station_meta()
    meta_name2coord = {
        str(row.station).strip(): (float(row.lat), float(row.lon), float(row.elevation))
        for row in meta.itertuples()
    }

    rows = []
    csv_files = sorted(RAW3H.glob("*_3h.csv"))
    print(f"[{run_id}] found {len(csv_files)} station csv files")

    for f in csv_files:
        station = f.stem.replace("_3h", "").strip()
        df = load_station_csv(f)
        n_total = len(df)

        # 时间解析失败的行直接丢弃前先计数
        try:
            ts = pd.to_datetime(
                dict(year=df.year, month=df.month, day=df.day, hour=df.hour),
                errors="coerce",
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(dict(station=station, file=f.name, n_total=n_total,
                             error=str(exc)))
            continue
        valid_time = ts.notna()
        df = df.assign(time=ts)[valid_time].sort_values("time")

        n_ws = int(df.wspd.notna().sum())
        n_wd = int(df.wdir.notna().sum())
        integ_ws = n_ws / n_total if n_total else 0.0
        integ_wd = n_wd / n_total if n_total else 0.0

        # 与 GFS 窗口重叠 (按有风速观测的年份计)
        yrs = df.loc[df.wspd.notna(), "year"]
        yrs_overlap = yrs[(yrs >= GFS_MIN_YEAR) & (yrs <= GFS_MAX_YEAR)]
        n_overlap_years = yrs_overlap.nunique()

        # 基本气候统计 (全时段)
        ws = df.wspd.dropna()
        clim = dict(
            ws_mean=float(ws.mean()) if len(ws) else np.nan,
            ws_p95=float(ws.quantile(0.95)) if len(ws) else np.nan,
            ws_max=float(ws.max()) if len(ws) else np.nan,
        )

        lat, lon, elev = meta_name2coord.get(station, (np.nan, np.nan, np.nan))
        rows.append(dict(
            station=station, file=f.name, lat=lat, lon=lon, elevation=elev,
            n_total=n_total, n_ws=n_ws, n_wd=n_wd,
            integ_ws=round(integ_ws, 4), integ_wd=round(integ_wd, 4),
            overlap_years_2015_2021=int(n_overlap_years),
            t_start=str(df.time.min()), t_end=str(df.time.max()),
            **clim,
        ))

    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "station_inventory.csv", index=False)

    usable = inv[
        (inv.integ_ws >= MIN_WIND_INTEGRITY)
        & (inv.overlap_years_2015_2021 >= MIN_OVERLAP_YEARS)
    ]
    usable.to_csv(OUT / "usable_stations.csv", index=False)

    # --- 运行自检记录 ---
    selfcheck = dict(
        run_id=run_id,
        started_utc=t0.isoformat(),
        finished_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha256(Path(__file__).resolve()),
        inputs=dict(
            n_station_csv=len(csv_files),
            shapefile_sha256=sha256(SHP.with_suffix(".shp")) if SHP.exists() else None,
        ),
        criteria=dict(min_wind_integrity=MIN_WIND_INTEGRITY,
                      overlap_years=MIN_OVERLAP_YEARS,
                      window=f"{GFS_MIN_YEAR}-{GFS_MAX_YEAR}"),
        results=dict(
            n_inventory=len(inv),
            n_usable=int(len(usable)),
            n_usable_integ_only=int((inv.integ_ws >= MIN_WIND_INTEGRITY).sum()),
            n_usable_overlap_only=int((inv.overlap_years_2015_2021 >= MIN_OVERLAP_YEARS).sum()),
        ),
    )
    with open(OUT / "e1_qc_run.json", "w", encoding="utf-8") as f:
        json.dump(selfcheck, f, ensure_ascii=False, indent=2)

    print(f"[{run_id}] inventory={len(inv)} usable={len(usable)} "
          f"(integ>=50%: {selfcheck['results']['n_usable_integ_only']}, "
          f"overlap>=3y: {selfcheck['results']['n_usable_overlap_only']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
