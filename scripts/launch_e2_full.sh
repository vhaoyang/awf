#!/bin/bash
# AWF 启动器: 2021 有档日 GFS 下载 (dm 环境绝对路径, setsid 脱离会话, 断点续传)
export HOME=/home/vhyang
PY=/home/vhyang/miniconda3/envs/dm/bin/python
cd /home/vhyang/zcode/awf
exec setsid nohup "$PY" scripts/e2_gfs_download.py \
  --days-file /mnt/d/data/awfdata/meta/gfs_2021_available_days.txt \
  --cycle 00 \
  --leads 0 6 12 24 48 72 \
  >> logs/e2_gfs_full2021.log 2>&1 < /dev/null
