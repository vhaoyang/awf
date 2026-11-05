#!/usr/bin/env python
"""Build the station institution and nominal sensor-height registry.

Institution names come from Table S1 of the AntAWS source publication,
with five explicit name matches below. Heights are network-level nominal
values, not station-specific measurements. They do not resolve changes due
to snow accumulation or establish a uniform height-induced bias.
The source table is an external input; see docs/REPRODUCE.md.
Run ID: AWF_E1B_REGISTRY_R1."""
import os
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA = Path(os.environ.get("AWF_DATA_ROOT", "data"))
SUPP = Path(os.environ.get("AWF_SENSOR_TABLE", str(DATA / "meta/essd_supp.txt")))
ST = DATA / "derived" / "station_qc"
OUT_CSV = ST / "station_sensor_registry.csv"
OUT_META = ST / "station_sensor_registry_meta.json"

# Table S1 正则匹配不到的站 (整型经纬度等原因), 人工核对原文后登记
MANUAL = {"Emma": "AMRC", "Kathie": "AMRC", "Nico": "AMRC",
          "Panda300": "CHINARE", "aws09": "IMAU"}
# 标称高度: 论文正文 "~3 m typical, 中山 10 m"; 其余网络按 3 m 标称
NOMINAL_H = {"AMRC": 3.0, "POLENET": 3.0, "PNRA": 3.0, "IMAU": 3.0,
             "CHINARE": 3.0, "AAD": 3.0, "Zhongshan": 10.0}
EXCEPT_10M = {"Zhongshan"}

ELEV_BINS = [-1, 100, 500, 1000, 2000, 4500]
ELEV_LABELS = ["coast<100", "100-500", "500-1000", "slope1-2k", "plateau>2k"]


def parse_s1() -> dict:
    recs = {}
    for ln in SUPP.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\s*(\d{1,3})\s+(\S.*?)\s+(-?\d+(?:\.\d+)?)\s+"
                     r"(-?\d+(?:\.\d+)?)\s+(\d+)\s+(\S+)\s*$", ln)
        if m:
            name = m.group(2).strip()
            if name not in recs:               # 首次出现为准 (S1 按序号)
                recs[name] = m.group(6).strip()
    return recs


def main() -> int:
    s1 = parse_s1()
    st = pd.read_csv(ST / "usable_stations.csv")
    st["institution"] = st.station.map(s1).fillna(st.station.map(MANUAL))
    assert st.institution.notna().all(), st[st.institution.isna()].station.tolist()

    st["nominal_anem_height_m"] = st.apply(
        lambda r: 10.0 if r.station in EXCEPT_10M
        else NOMINAL_H.get(r.institution, 3.0), axis=1)
    st["terrain"] = pd.cut(st.elevation, ELEV_BINS, labels=ELEV_LABELS)
    cols = ["station", "lat", "lon", "elevation", "terrain",
            "institution", "nominal_anem_height_m"]
    st[cols].sort_values("station").to_csv(OUT_CSV, index=False)

    xtab = pd.crosstab(st.terrain, st.institution)
    meta = dict(
        run_id="AWF_E1B_REGISTRY_R1",
        utc=datetime.now(timezone.utc).isoformat(),
        n_stations=len(st),
        sources={
            "institution": "ESSD 15,411-429 suppl. Table S1 "
                           "(pdftotext -layout 解析 + 5 站人工核对)",
            "nominal_height": "论文正文: 传感器 1-6 m, 风标称 ~3 m, "
                              "中山站 10 m; 网络级标称非逐站实测",
            "unmatched_manual": MANUAL,
        },
        height_caveat=("Heights are nominal network-level metadata, not station-specific "
                       "measurements. Snow accumulation and sensor differences remain "
                       "unresolved. Neutral log-profile scenarios do not establish a "
                       "uniform bias or eliminate terrain-dependent confounding."),
        terrain_x_institution=xtab.to_dict(),
    )
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(st.institution.value_counts().to_string())
    print(xtab.to_string())
    print(f"-> {OUT_CSV}\n-> {OUT_META}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
