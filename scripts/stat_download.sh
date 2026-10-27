#!/bin/bash
n=$(ls /mnt/d/data/awfdata/derived/paired/gfs_point/ | wc -l)
json=$(grep -c '^{' /home/vhyang/zcode/awf/logs/e2_gfs_full2021.log)
pid=$(pgrep -f 'e2_gfs_download.py' | head -1)
et=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
echo "done_days=$n json_lines=$json elapsed_s=$et"
