#!/bin/bash
DATA_ROOT="${AWF_DATA_ROOT:-data}"
n=$(ls "$DATA_ROOT"/derived/paired/gfs_point/ | wc -l)
json=$(grep -c '^{' logs/e2_gfs_full2021.log)
pid=$(pgrep -f 'e2_gfs_download.py' | head -1)
et=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
echo "done_days=$n json_lines=$json elapsed_s=$et"
