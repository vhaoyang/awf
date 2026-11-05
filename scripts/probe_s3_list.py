#!/usr/bin/env python
"""Inspect S3 prefixes and sample keys for GFS archive discovery.

Counts returned without pagination describe the returned page, not necessarily
the complete prefix. Use paginated listing for a complete inventory."""
import xml.etree.ElementTree as ET
from urllib.parse import quote
from urllib.request import urlopen, Request

NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
BUCKET = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"


def list_keys(prefix: str, max_keys: int = 8) -> tuple[int, list[str]]:
    """Return the count and sample keys on one S3 result page.
    
        KeyCount is a page count; this helper does not paginate a truncated listing."""
    url = f"{BUCKET}?list-type=2&prefix={quote(prefix)}&max-keys={max_keys}"
    with urlopen(Request(url, headers={"User-Agent": "awf/0.1"}), timeout=60) as r:
        tree = ET.parse(r)
    root = tree.getroot()
    n = int(root.findtext(f"{NS}KeyCount", "0"))
    keys = [e.findtext(f"{NS}Key") for e in root.findall(f"{NS}Contents")]
    return n, keys


if __name__ == "__main__":
    # 老桶 noaa-gfs-pds 各年覆盖
    BUCKETS = ["https://noaa-gfs-pds.s3.amazonaws.com",
               "https://noaa-gfs-bdp-pds.s3.amazonaws.com"]
    globals()["BUCKET"] = BUCKETS[0]
    print("=== old bucket noaa-gfs-pds ===")
    for y in range(2015, 2022):
        try:
            n, keys = list_keys(f"gfs.{y}0101/", max_keys=4)
            print(y, "objects:", n, keys[:2])
        except Exception as e:  # noqa: BLE001
            print(y, "ERR", str(e)[:80])
    print("=== new bucket 2021 monthly coverage ===")
    globals()["BUCKET"] = BUCKETS[1]
    for m in range(1, 13):
        ymd = f"2021{m:02d}01"
        try:
            n, keys = list_keys(f"gfs.{ymd}/", max_keys=4)
            print(ymd, "objects:", n, keys[:1])
        except Exception as e:  # noqa: BLE001
            print(ymd, "ERR", str(e)[:80])
