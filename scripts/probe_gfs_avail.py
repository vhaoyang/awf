#!/usr/bin/env python
r"""AWF 辅助: 探测 GFS 存档各年可得性 (非任务号脚本)

老档文件命名可能有差异, 对每年测多个候选:
  - gfs.t00z.pgrb2.0p25.f000  (2021 实测存在)
  - gfs.t00z.pgrb2full0.25.f000
  - gfs.t00z.pgrb2f000 (0.5度时代的名字)
并对每候选测 idx。
"""
from urllib.request import Request, urlopen
from urllib.error import HTTPError

BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
CANDS = [
    "gfs.{ymd}/{cc}/atmos/gfs.t{cc}z.pgrb2.0p25.f{fff}.idx",
    "gfs.{ymd}/{cc}/atmos/gfs.t{cc}z.pgrb2full0.25.f{fff}.idx",
    "gfs.{ymd}/{cc}/gfs.t{cc}z.pgrb2f{fff}.idx",
]


def probe(ymd: str, cc: str = "00", fff: str = "000") -> str:
    for tpl in CANDS:
        url = f"{BASE}/" + tpl.format(ymd=ymd, cc=cc, fff=fff)
        try:
            req = Request(url, method="HEAD", headers={"User-Agent": "awf/0.1"})
            with urlopen(req, timeout=30) as r:
                if r.status == 200:
                    return "OK " + url
        except HTTPError as e:
            if e.code != 404:
                return f"ERR{e.code} " + url
        except Exception as e:  # noqa: BLE001
            return f"ERR " + url
    return "404"


if __name__ == "__main__":
    # 每年测 1 号和年中两个日期, 减少单日缺档误判
    tests = []
    for y in range(2015, 2022):
        tests += [f"{y}0101", f"{y}0701", f"{y}1231"]
    for ymd in tests:
        print(ymd, "->", probe(ymd), flush=True)
