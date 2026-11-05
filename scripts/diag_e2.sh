#!/bin/bash
DATA_ROOT="${AWF_DATA_ROOT:-data}"
# AWF 辅助: 诊断 e2 下载进程状态
echo "== pgrep =="
pgrep -af e2_gfs_download
echo "== ps =="
ps aux | grep -v grep | grep e2_gfs_download
echo "== log tail =="
tail -c 400 logs/e2_gfs_full2021.log
echo
echo "== files =="
ls "$DATA_ROOT"/derived/paired/gfs_point/ | wc -l
