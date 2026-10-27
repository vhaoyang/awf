#!/usr/bin/env python
"""AWF E1B — 站点传感器/机构登记表（风速计高度混杂评估的前置）

运行号: AWF_E1B_REGISTRY_R1 (服务器端自编, 非本侧任务号)

来源:
  ESSD 15, 411-429 (Wang et al. 2023) 附录 Table S1 (逐站机构, pdftotext 解析)
  正文: 传感器高度 1-6 m、风观测标称 ~3 m、中山站 10 m; 无逐站精确高度,
        且高度随积雪掩埋漂移(时变, 无公开数据)。

输出:
  derived/station_qc/station_sensor_registry.csv — 74 可用站 + 机构 + 标称高度
  derived/station_qc/station_sensor_registry_meta.json — 口径与来源登记

已知限制 (E4 解释必读):
  nominal_anem_height_m 是网络级标称值, 不是逐站实测; 高度错配的主要部分
  (obs ~3m vs GFS 10m) 是准均匀偏移, 进总偏差不进条件结构; 残余混杂=
  积雪掩埋漂移(时变逐站, 不可得) 与 少数非 3m 站。机构 x 地形交叉表
  已核: 每个地形档 2-4 机构混合, 机构混杂对地形分层影响有限(高原档
  CHINARE 偏多, 见 meta json)。
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SUPP = Path("/tmp/essd_supp.txt")   # pdftotext -layout 的 ESSD 附录
DATA = Path("/mnt/d/data/awfdata")
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
        height_caveat=("逐站精确高度不存在于公开元数据; 积雪掩埋漂移时变不可得。"
                       "obs~3m vs GFS 10m 的对数廓线错配预期 +15~25% 正偏差"
                       "(z0=0.01~0.1 m), 属准均匀偏移进总偏差不进条件结构。"),
        terrain_x_institution=xtab.to_dict(),
    )
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(st.institution.value_counts().to_string())
    print(xtab.to_string())
    print(f"-> {OUT_CSV}\n-> {OUT_META}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
