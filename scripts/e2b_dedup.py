#!/usr/bin/env python
"""AWF E2B — E2 产物去重修复（在 E2 全量下载结束之后、E3 配对之前运行）

运行号: AWF_E2B_DEDUP_R1 (服务器端自编, 非本侧任务号)

背景: e2_gfs_download.py 的断点续传在重跑被中断的天时会重复追加全部 lead
(实测 20210326/20210426/20211231 受影响), E3 的 one_to_many 合并会静默吸收
重复行使该天双倍加权。本脚本对 gfs_point/*.parquet 按
(station, valid_time, lead_h) 去重并校验 lead 完整性, 修复前自动备份。

用法: python scripts/e2b_dedup.py [--dry-run]
"""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA = Path("/mnt/d/data/awfdata")
GFS = DATA / "derived" / "paired" / "gfs_point"
LEADS = [0, 6, 12, 24, 48, 72]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    report = dict(
        run_id="AWF_E2B_DEDUP_R1",
        utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dry_run=a.dry_run, files=[], n_removed_total=0,
    )
    backup = GFS.parent / "gfs_point_prededup_backup"
    if not a.dry_run:
        backup.mkdir(exist_ok=True)

    for f in sorted(GFS.glob("gfs_*.parquet")):
        df = pd.read_parquet(f)
        n0 = len(df)
        leads = sorted(df.lead_h.unique())
        ded = df.drop_duplicates(subset=["station", "valid_time", "lead_h"],
                                 keep="first")
        n1 = len(ded)
        leads_ok = leads == LEADS
        row = dict(file=f.name, rows=n0, dups_removed=n0 - n1,
                   leads_complete=bool(leads_ok),
                   stations=int(ded.station.nunique()))
        if n0 != n1 and not a.dry_run:
            if not (backup / f.name).exists():
                shutil.copy2(f, backup / f.name)
            ded.to_parquet(f, index=False)
        report["files"].append(row)
        report["n_removed_total"] += n0 - n1
        flag = "" if (leads_ok and n0 == n1) else "  <-- FIX" + \
            ("ED" if not a.dry_run else " NEEDED")
        print(f"{f.name}: rows={n0} dups={n0-n1} leads_ok={leads_ok}"
              f" stations={ded.station.nunique()}{flag}")

    n_bad_leads = sum(1 for r in report["files"] if not r["leads_complete"])
    report["n_files"] = len(report["files"])
    report["n_files_incomplete_leads"] = n_bad_leads
    out = Path("/home/vhyang/zcode/awf/outputs/e2b_dedup_report.json")
    if not a.dry_run:
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\nremoved_total={report['n_removed_total']} "
              f"incomplete_lead_files={n_bad_leads} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
