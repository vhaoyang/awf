#!/bin/bash
DATA_ROOT="${AWF_DATA_ROOT:-data}"
# 对比有档日清单与实际产出文件, 找缺日
diff <(sed 's/^/gfs_/;s/$/_00z.parquet/' "$DATA_ROOT"/meta/gfs_2021_available_days.txt | sort) \
     <(ls "$DATA_ROOT"/derived/paired/gfs_point/ | sort) | head -10
