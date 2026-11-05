#!/bin/bash
DATA_ROOT="${AWF_DATA_ROOT:-data}"
# AWF 启动器: 2021 有档日 GFS 下载 (setsid 脱离会话, 断点续传)
# run from repo root
PY=python
cd "$(dirname "$0")/.."
exec setsid nohup "$PY" scripts/e2_gfs_download.py \
  --days-file "$DATA_ROOT"/meta/gfs_2021_available_days.txt \
  --cycle 00 \
  --leads 0 6 12 24 48 72 \
  >> logs/e2_gfs_full2021.log 2>&1 < /dev/null
