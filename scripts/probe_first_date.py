#!/usr/bin/env python
r"""AWF 辅助: 二分查 noaa-gfs-bdp-pds 桶内 pgrb2.0p25 文件的实际起始日期 (非任务号脚本)"""
import xml.etree.ElementTree as ET
from urllib.parse import quote
from urllib.request import urlopen, Request

NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"


def has_pgrb2(ymd: str) -> bool:
    prefix = f"gfs.{ymd}/00/atmos/gfs.t00z.pgrb2.0p25.f000"
    url = f"{BUCKET}?list-type=2&prefix={quote(prefix)}&max-keys=2"
    try:
        with urlopen(Request(url, headers={"User-Agent": "awf/0.1"}), timeout=40) as r:
            tree = ET.parse(r)
        return int(tree.getroot().findtext(f"{NS}KeyCount", "0")) > 0
    except Exception:  # noqa: BLE001
        return False


def bisect(start: str, end: str):
    """start 无档(假设), end 有档(假设), 二分找最早有档日。"""
    from datetime import date, timedelta
    lo = date.fromisoformat(start)   # 无档
    hi = date.fromisoformat(end)     # 有档
    while (hi - lo).days > 1:
        mid = lo + (hi - lo) / 2
        if has_pgrb2(mid.isoformat()):
            hi = mid
        else:
            lo = mid
        print("  range:", lo, "..", hi, flush=True)
    return hi


if __name__ == "__main__":
    print("20210101:", has_pgrb2("2021-01-01".replace("-", "")))
    print("20210601:", has_pgrb2("20210601"))
    print("20210701:", has_pgrb2("20210701"))
    print("20211231:", has_pgrb2("20211231"))
    print("first available:", bisect("2021-01-01", "2021-12-31"))
