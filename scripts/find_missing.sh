#!/bin/bash
# 对比有档日清单与实际产出文件, 找缺日
diff <(sed 's/^/gfs_/;s/$/_00z.parquet/' /mnt/d/data/awfdata/meta/gfs_2021_available_days.txt | sort) \
     <(ls /mnt/d/data/awfdata/derived/paired/gfs_point/ | sort) | head -10
