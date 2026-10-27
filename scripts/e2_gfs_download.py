#!/usr/bin/env python
r"""AWF E2_GFS — GFS 0.25° 历史预报按需分段下载 + 站点提取（S1 阶段第二步）

运行号: AWF_E2_GFS_R1 (服务器端自编, 非本侧任务号)

原理:
  NOAA AWS 开放数据桶 noaa-gfs-bdp-pds 存有 2015 起 GFS 0.25° 全量 GRIB2
  (~500 MB/文件, 不可整档下载)。每个文件伴随 .idx 线索引, 记录逐 GRIB 消息的
  变量名/层/字节偏移。本脚本解析 .idx 后用 HTTP Range 只取需要的消息
  (U10M/V10M/TMP2m above ground/PRMSL/TSOIL 等, 每文件 2–6 MB),
  cfgrib 解码 -> 双线性插值到 74 站 -> 追加 parquet, 临时 GRIB 即删。

  单日单 cycle 单 lead 网络量 ~5 MB; 全量计划 (00Z, leads 0/6/12/24/48/72,
  2015-01-01 ~ 2021-12-31) ≈ 2557 天 x 6 文件 x 5 MB ≈ 77 GB 网络流量,
  落盘仅站点级 parquet (~百 MB 级)。

用法:
  python e2_gfs_download.py --date 2021-12-31 --cycle 00 --leads 0   # 冒烟测试
  python e2_gfs_download.py --start 2015-01-01 --end 2021-12-31      # 全量(后台分批)
"""
import argparse
import hashlib
import io
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import xarray as xr

DATA = Path("/mnt/d/data/awfdata")
RAW_GFS = DATA / "raw" / "gfs"
PAIRED = DATA / "derived" / "paired"
STATION_FILE = DATA / "derived" / "station_qc" / "usable_stations.csv"

BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
# GRIB 消息短名 (与 .idx 第4列对应): 变量 -> cfgrib shortName
WANTED = {
    "UGRD": ("10 metre U wind component", "u10"),
    "VGRD": ("10 metre V wind component", "v10"),
    "TMP": ("2 metre temperature", "t2m"),
    "PRMSL": ("Pressure reduced to MSL", "prmsl"),
    "RH": ("Relative humidity", "rh2m"),  # 2 above ground
}
LVL_HINT = {"UGRD": "10 m", "VGRD": "10 m", "TMP": "2 m", "PRMSL": "", "RH": "2 m"}

SLEEP_S = 2.0          # 请求间隔, 对 S3 礼貌
RETRY = 4


def http_get(url: str, rng: str | None = None, timeout: int = 120) -> bytes:
    hdr = {"User-Agent": "awf-research/0.1"}
    if rng:
        hdr["Range"] = rng
    last = None
    for i in range(RETRY):
        try:
            with urlopen(Request(url, headers=hdr), timeout=timeout) as r:
                return r.read()
        except HTTPError as e:
            if e.code in (404, 403):
                raise
            last = e
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(SLEEP_S * (2 ** i))
    raise RuntimeError(f"GET failed {url}: {last}")


def parse_idx(idx_text: str) -> list[dict]:
    """解析 wgrib2 风格 .idx: 消息号:起始字节:日期:变量:层:类型:短名:..."""
    msgs = []
    for line in idx_text.splitlines():
        parts = line.split(":")
        if len(parts) >= 6:
            msgs.append(dict(
                msg=int(parts[0]), start=int(parts[1]),
                var=parts[3], lvl=":".join(parts[4:6]).strip(" "), name=parts[-1].strip(),
            ))
    return msgs


def fetch_subset(date: str, cycle: int, lead: int, tmpdir: Path) -> Path | None:
    """下载单文件的所需消息, 拼成小 GRIB 返回路径; 文件不存在返回 None。"""
    d = datetime.strptime(date, "%Y-%m-%d")
    fdir = f"gfs.{d:%Y%m%d}/{cycle:02d}/atmos"
    fstem = f"gfs.t{cycle:02d}z.pgrb2.0p25.f{lead:03d}"
    url_grib = f"{BASE}/{fdir}/{fstem}"
    idx = http_get(f"{url_grib}.idx").decode("ascii", "replace")
    msgs = parse_idx(idx)
    total_len = msgs[-1]["start"] + 200_000_000  # idx 只记到末消息起点

    # 选中消息区间: [start_i, start_{i+1}) — idx 里下一行起点即本消息终点
    sel, ranges = [], []
    for i, m in enumerate(msgs):
        nxt = msgs[i + 1]["start"] if i + 1 < len(msgs) else None
        lvl = m["lvl"]
        ok = False
        if m["var"] in ("UGRD", "VGRD") and "10 m above ground" in lvl:
            ok = True
        if m["var"] == "TMP" and "2 m above ground" in lvl:
            ok = True
        if m["var"] == "RH" and "2 m above ground" in lvl:
            ok = True
        if m["var"] == "PRMSL" and "mean sea level" in lvl:
            ok = True
        if ok:
            sel.append(m)
            ranges.append((m["start"], nxt if nxt else m["start"] + 50_000_000))

    if not sel:
        return None
    buf = io.BytesIO()
    for (s, e) in ranges:
        chunk = http_get(url_grib, rng=f"bytes={s}-{e - 1}")
        buf.write(chunk)
        time.sleep(0.3)
    out = tmpdir / f"{fstem}.subset.grib2"
    out.write_bytes(buf.getvalue())
    return out


def interp_to_stations(g: Path, st: pd.DataFrame) -> pd.DataFrame:
    """cfgrib.open_datasets 自动分组(避免 heightAboveGround 键冲突),
    只取含目标地面变量的组, 各自插值后按站合并。"""
    import cfgrib
    dss = cfgrib.open_datasets(str(g), backend_kwargs={"indexpath": ""})
    want_cols = {"u10", "v10", "t2m", "r2", "prmsl"}
    parts = []
    for ds in dss:
        vars_ = set(ds.data_vars) & want_cols
        if not vars_:
            ds.close()
            continue
        lon = ds.longitude.values
        lon = np.where(lon > 180, lon - 360, lon)
        ds = ds.assign_coords(longitude=lon).sortby("longitude")
        pts = ds.interp(
            latitude=xr.DataArray(st.lat.values, dims="station"),
            longitude=xr.DataArray(st.lon.values, dims="station"),
            method="linear",
        )
        df = pts.to_dataframe().reset_index()
        df = df.assign(station=st.station.values)
        cols = ["station", "latitude", "longitude"] + sorted(vars_) \
            if parts == [] else ["station"] + sorted(vars_)
        parts.append(df[cols])
        ds.close()

    out = parts[0]
    for p in parts[1:]:
        out = pd.merge(out, p, on="station", how="outer")
    return out.rename(columns={"r2": "rh2m"})


def ds_time(g: Path):
    """读小 GRIB 的基准时间(从任一成功组取 time)。"""
    import cfgrib
    dss = cfgrib.open_datasets(str(g), backend_kwargs={"indexpath": ""})
    t = None
    for ds in dss:
        if "time" in ds.coords:
            t = pd.Timestamp(ds.time.values)
            ds.close()
            break
        ds.close()
    if t is None:
        raise RuntimeError("no time coord in subset grib")
    return t


def date_done(outdir: Path, date: str, cycle: int, leads: list[int]) -> bool:
    """断点续传: 该日 parquet 已含全部 lead 则跳过。"""
    pq = outdir / f"gfs_{date.replace('-', '')}_{cycle:02d}z.parquet"
    if not pq.exists():
        return False
    try:
        old = pd.read_parquet(pq, columns=["lead_h"])
        return set(old.lead_h.unique()) >= set(leads)
    except Exception:  # noqa: BLE001
        return False


def process_day(date: str, cycle: int, leads: list[int], st: pd.DataFrame) -> dict:
    tmpdir = RAW_GFS / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    outdir = PAIRED / "gfs_point"
    outdir.mkdir(parents=True, exist_ok=True)
    stat = dict(date=date, cycle=cycle, leads_done=[], bytes=0, skipped=False)
    if date_done(outdir, date, cycle, leads):
        stat["skipped"] = True
        return stat
    from concurrent.futures import ThreadPoolExecutor

    def _one(lead: int):
        """下载单个 lead 的 subset 文件, 返回 (lead, path|None|'404')。"""
        try:
            g = fetch_subset(date, cycle, lead, tmpdir)
        except HTTPError as e:
            if e.code == 404:
                return (lead, "404")
            raise
        return (lead, g)

    with ThreadPoolExecutor(max_workers=6) as ex:
        results = dict(ex.map(_one, leads))

    for lead in leads:
        g = results.get(lead)
        if g == "404" or g is None:
            stat["leads_done"].append((lead, "404" if g == "404" else "empty"))
            continue
        df = interp_to_stations(g, st)
        df = df.assign(valid_time=pd.Timestamp(ds_time(g)) + pd.Timedelta(hours=lead),
                       lead_h=lead, init=date + f" {cycle:02d}Z")
        pq = outdir / f"gfs_{date.replace('-', '')}_{cycle:02d}z.parquet"
        if pq.exists():
            old = pd.read_parquet(pq)
            df = pd.concat([old, df], ignore_index=True)
        df.to_parquet(pq, index=False)
        stat["bytes"] += g.stat().st_size
        stat["leads_done"].append((lead, f"{len(df)}rows"))
        g.unlink()
        time.sleep(SLEEP_S)
    return stat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")            # 单日冒烟: 2021-12-31
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--days-file", help="只跑清单内的日期 (YYYYMMDD/行), 优先于 start/end")
    ap.add_argument("--cycle", type=int, default=0)
    ap.add_argument("--leads", type=int, nargs="+", default=[0, 6, 12, 24, 48, 72])
    a = ap.parse_args()

    st = pd.read_csv(STATION_FILE)
    run = dict(run_id="AWF_E2_GFS_R1", started=datetime.now(timezone.utc).isoformat(),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dates = []
    if a.days_file:
        raw = [ln.strip().replace("-", "") for ln in Path(a.days_file).read_text().splitlines()
               if ln.strip()]
        dates = [f"{s[:4]}-{s[4:6]}-{s[6:]}" for s in raw]
    elif a.date:
        dates = [a.date]
    elif a.start and a.end:
        d0 = datetime.strptime(a.start, "%Y-%m-%d"); d1 = datetime.strptime(a.end, "%Y-%m-%d")
        while d0 <= d1:
            dates.append(d0.strftime("%Y-%m-%d")); d0 += timedelta(days=1)

    n_err = 0
    for date in dates:
        try:
            s = process_day(date, a.cycle, a.leads, st)
        except Exception as e:  # noqa: BLE001  单日失败不杀全程
            n_err += 1
            print(json.dumps({"date": date, "fatal": str(e)[:200]}), flush=True)
            continue
        print(json.dumps(s), flush=True)

    run["finished"] = datetime.now(timezone.utc).isoformat()
    (PAIRED / "e2_gfs_run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
