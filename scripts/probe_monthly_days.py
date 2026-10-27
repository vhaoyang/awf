#!/usr/bin/env python
r"""AWF 辅助: 按月清点 pgrb2.0p25 实际有档天数 (非任务号脚本)"""
import xml.etree.ElementTree as ET
from urllib.parse import quote
from urllib.request import urlopen, Request

NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"


def list_days_with_f000(month: str) -> list[str]:
    """month='202101', 返回该月有 gfs.t00z.pgrb2.0p25.f000 的日期键(分页)。"""
    import time
    days = []
    token = None
    for attempt in range(6):
        try:
            while True:
                url = (f"{BUCKET}?list-type=2&prefix=gfs.{month}"
                       f"&max-keys=300")
                if token:
                    url += f"&continuation-token={quote(token)}"
                with urlopen(Request(url, headers={"User-Agent": "awf/0.1"}),
                             timeout=90) as r:
                    tree = ET.parse(r)
                root = tree.getroot()
                for e in root.findall(f"{NS}Contents"):
                    k = e.findtext(f"{NS}Key")
                    if k.endswith("/atmos/gfs.t00z.pgrb2.0p25.f000"):
                        days.append(k.split("/")[0].replace("gfs.", ""))
                token = root.findtext(f"{NS}NextContinuationToken")
                if not token:
                    return sorted(set(days))
                time.sleep(1.0)
        except Exception:  # noqa: BLE001
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"list failed for {month} after retries")


if __name__ == "__main__":
    total = 0
    for m in range(1, 13):
        mm = f"2021{m:02d}"
        days = list_days_with_f000(mm)
        total += len(days)
        print(mm, "days_with_f000:", len(days),
              ("first=" + days[0] + " last=" + days[-1]) if days else "")
    print("TOTAL 2021 days:", total)
