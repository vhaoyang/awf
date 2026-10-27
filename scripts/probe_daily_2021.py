#!/usr/bin/env python
r"""AWF 辅助: 逐日紧前缀探测 2021 年 pgrb2.0p25.f000 有档日清单 (非任务号脚本)

每天只查一个对象的紧前缀, 快且省。输出可用日清单到 meta/。
"""
import json
import time
from datetime import date, timedelta
from urllib.parse import quote
from urllib.request import urlopen, Request

NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
OUT = "/mnt/d/data/awfdata/meta/gfs_2021_available_days.json"


def day_has(ymd: str) -> bool:
    prefix = f"gfs.{ymd}/00/atmos/gfs.t00z.pgrb2.0p25.f000"
    url = f"{BUCKET}?list-type=2&prefix={quote(prefix)}&max-keys=1"
    for attempt in range(5):
        try:
            with urlopen(Request(url, headers={"User-Agent": "awf/0.1"}),
                         timeout=30) as r:
                tree = ET.parse(r)
            return int(tree.getroot().findtext(f"{NS}KeyCount", "0")) > 0
        except Exception:  # noqa: BLE001
            time.sleep(1.5 * (attempt + 1))
    return False


if __name__ == "__main__":
    d0 = date(2021, 1, 1)
    d1 = date(2021, 12, 31)
    days = []
    cur = d0
    while cur <= d1:
        ymd = cur.strftime("%Y%m%d")
        if day_has(ymd):
            days.append(ymd)
        if cur.day % 30 == 0 or cur == d1:
            print(f"... {ymd}: cum {len(days)} days", flush=True)
        cur += timedelta(days=1)
        time.sleep(0.4)

    with open(OUT, "w") as f:
        json.dump(dict(n_days=len(days), days=days), f)
    print("TOTAL:", len(days))
    # 月份分布
    from collections import Counter
    c = Counter(d[:6] for d in days)
    for k in sorted(c):
        print(k, c[k])
