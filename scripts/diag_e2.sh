#!/bin/bash
# AWF 辅助: 诊断 e2 下载进程状态
echo "== pgrep =="
pgrep -af e2_gfs_download
echo "== ps =="
ps aux | grep -v grep | grep e2_gfs_download
echo "== log tail =="
tail -c 400 /home/vhyang/zcode/awf/logs/e2_gfs_full2021.log
echo
echo "== files =="
ls /mnt/d/data/awfdata/derived/paired/gfs_point/ | wc -l
