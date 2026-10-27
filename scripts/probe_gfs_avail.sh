#!/bin/bash
# AWF 辅助: 探测 GFS 0.25° 存档各年可得性 (非任务号脚本)
for y in 2015 2016 2017 2018 2019 2020 2021; do
  code=$(curl -s -o /dev/null -w "%{http_code}" -I "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.${y}0101/00/atmos/gfs.t00z.pgrb2.0p25.f000.idx")
  echo "$y: $code"
done
