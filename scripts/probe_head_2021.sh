#!/bin/bash
# AWF 辅助: curl HEAD 逐日探测 2021 pgrb2.0p25.f000 有档日 (普通 GET 通道快, S3 API 通道被网络拖死)
# 输出: /mnt/d/data/awfdata/meta/gfs_2021_available_days.txt  (YYYYMMDD 每行一个)
OUT=/mnt/d/data/awfdata/meta/gfs_2021_available_days.txt
: > "$OUT"
for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
  n=0
  for d in $(seq -w 1 31); do
    ymd="2021${m}${d}"
    # 跳过不存在的日期 (2月30等)
    python3 -c "import datetime;datetime.date(2021,${m#0},${d#0})" 2>/dev/null || continue
    code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 8 -m 15 \
      -I "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.${ymd}/00/atmos/gfs.t00z.pgrb2.0p25.f000.idx")
    if [ "$code" = "200" ]; then
      echo "$ymd" >> "$OUT"
      n=$((n+1))
    fi
  done
  echo "2021-${m}: ${n} days ok" >&2
done
wc -l "$OUT" >&2
