#!/usr/bin/env python
"""Deduplicate daily GFS station forecasts and report file completeness.

The utility preserves pre-edit files as implemented below. Use --dry-run to
inspect proposed changes before rewriting a local working copy.
Run ID: AWF_E2B_DEDUP_R1."""
import os
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA = Path(os.environ.get("AWF_DATA_ROOT", "data"))
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
    out = Path("outputs/e2b_dedup_report.json")
    if not a.dry_run:
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\nremoved_total={report['n_removed_total']} "
              f"incomplete_lead_files={n_bad_leads} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
