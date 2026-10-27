#!/usr/bin/env python
"""AWF E10 — 第六轮（BQN+LightGBM+QRF）阶段1：协议冻结（E01/E02 部分）

运行号: AWF_E10_PROTOCOL_R1
依据: 05_投稿与归档/评审与修改/2026-10-24_WAF_manuscript_ams/10_统一执行方案 §三/§四/§五

产出(写入 outputs/bqn_lgbm_qrf/protocol/ 与交付包 AWF_BQN_LGBM_QRF_R1/protocol/):
  data_audit.json    数量与键核验(57,700/29,216/50/18,235/42/159/546)
  splits_summary.json T/S/F 三设计划分定义、行数、内层验证站表(确定性站ID哈希<20%)
  feature_manifest.json F0/F1 共同输入清单(实际列名)与边界(不含站ID/站史)
  candidates.json    T 设计预登记候选(BQN8/QRF8/LGBM8)与 S/F 预设
  config_mix.json    种子{17,42,2026}与等权CDF混合规则
  protocol.md        冻结说明与资源估计
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/mnt/d/data/awfdata")
PAIR = DATA / "derived" / "paired" / "paired_2021.parquet"
REG = DATA / "derived" / "station_qc" / "station_sensor_registry.csv"
OUT = Path("/home/vhyang/zcode/awf/outputs/bqn_lgbm_qrf/protocol")

SEEDS = [17, 42, 2026]
F0_FEATS = ["u10", "v10", "fcst_spd", "t2m", "prmsl", "rh2m", "lead_h",
            "month_sin", "month_cos", "lat", "lon", "elevation", "terrain"]
REMA_STATIC = ["slope1k", "slope5k", "slope25k", "prominence", "fetch_drop"]
F1_EXTRA = REMA_STATIC + ["katab_align_EN", "is_flat", "is_missing"]
CAND_BQN = [dict(degree=d, width=w, lr=lr)
            for d in (8, 12) for w in ((32, 16), (64, 32))
            for lr in (1e-3, 3e-4)]
CAND_QRF = [dict(min_samples_leaf=m, max_features=f)
            for m in (10, 30, 50, 100) for f in (0.7, 1.0)]
CAND_LGBM = [dict(num_leaves=l, min_data_in_leaf=m, learning_rate=r)
             for l in (15, 31) for m in (50, 100) for r in (0.03, 0.05)]


def station_hash20(s: str) -> bool:
    """确定性站 ID 哈希 <20% 选内层验证站(规则全局唯一)。"""
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % 100 < 20


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PAIR).rename(columns={"t2m_x": "obs_t2m",
                                               "t2m_y": "t2m"})
    reg = pd.read_csv(REG)
    df = df.merge(reg[["station", "lat", "lon", "elevation", "terrain"]],
                  on="station", how="left")
    df["terrain"] = df.terrain.astype("category")
    df["fcst_spd"] = np.hypot(df.u10, df.v10)
    df["month"] = df.valid_time.dt.month
    df["month_sin"] = np.sin(2 * np.pi * df.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.month / 12)
    n_miss = int(df.wspd.isna().sum())
    df = df.dropna(subset=["wspd"]).reset_index(drop=True)

    # ---- E01 数据核验 ----
    audit = dict(
        raw_paired_rows=74334, wspd_missing=n_miss, valid_rows=len(df),
        unique_station_valid=int(df.groupby(["station", "valid_time"]).ngroups),
        stations=int(df.station.nunique()),
        key_dupes=int(df.duplicated(["station", "init", "lead_h"]).sum()),
        test_rows=int((df.month >= 10).sum()),
        test_stations=int(df[df.month >= 10].station.nunique()),
        sep_init_test_rows=int(((df.month >= 10)
                                & (pd.to_datetime(df.init).dt.month == 9)).sum()),
        strong_events_test=int(((df.month >= 10) & (df.wspd > 15.0)).sum()),
        check=dict(
            rows_57700=len(df) == 57700,
            unique_29216=int(df.groupby(["station", "valid_time"]).ngroups) == 29216,
            stations_50=df.station.nunique() == 50,
            test_18235=int((df.month >= 10).sum()) == 18235,
            test_42=int(df[df.month >= 10].station.nunique()) == 42,
            sep_init_159=int(((df.month >= 10) & (pd.to_datetime(df.init).dt.month == 9)).sum()) == 159,
            events_546=int(((df.month >= 10) & (df.wspd > 15.0)).sum()) == 546,
            key_unique=df.duplicated(["station", "init", "lead_h"]).sum() == 0,
            valid_split_straddle_free=bool(
                ((df.month >= 10) | (df.month <= 9)).all())))
    assert all(bool(v) for v in audit["check"].values()), \
        f"数据核验失败: {audit['check']}"
    audit = json.loads(json.dumps(audit, default=lambda o: bool(o)
                                  if isinstance(o, np.bool_) else str(o)))
    (OUT / "data_audit.json").write_text(json.dumps(
        audit, ensure_ascii=False, indent=2))
    print("[audit]", json.dumps(audit["check"], ensure_ascii=False))

    # ---- 三设计划分 ----
    stations = sorted(df.station.unique())
    inner_val = [s for s in stations if station_hash20(s)]
    splits = dict(
        T=dict(rule="train valid_time Mar-Aug (data from Mar 23), val Sep, "
                    "test Oct-Dec; split on valid_time so all leads of a "
                    "valid_time stay in one part",
               train_rows=int((df.month <= 8).sum()),
               val_rows=int((df.month == 9).sum()),
               test_rows=int((df.month >= 10).sum()),
               sep_init_test_rows=audit["sep_init_test_rows"],
               disclosed="159 test rows with September inits kept"),
        S=dict(rule="50 LOO folds over stations, all months; held-out "
                    "station untouched; inner validation = deterministic "
                    "station-ID-hash 20% stations' records within allowed "
                    "window (their all-month rows from the 49 training "
                    "stations' side)",
               folds=50,
               inner_val_stations=inner_val,
               n_inner_val=len(inner_val),
               per_fold_train_rows=int(len(df) - df.groupby("station").size().mean()),
               note="exact per-fold counts in run manifests"),
        F=dict(rule="per station: train = other stations valid_time "
                    "<=2021-08-31 23:59, test = station valid_time "
                    ">=2021-10-01; September of other stations unused; "
                    "inner val = hash stations' August records",
               evaluable_rule="stations with >=50 test rows reported as "
                              "main (expected 42); others scored and "
                              "disclosed individually",
               all_stations=50))
    (OUT / "splits_summary.json").write_text(json.dumps(
        splits, ensure_ascii=False, indent=2))
    print("[splits] inner-val stations:", len(inner_val), inner_val[:6], "...")

    # ---- 特征清单 ----
    feat = dict(
        F0=dict(columns=F0_FEATS,
                note="identical inputs for all three models; no station ID, "
                     "no target-station history; terrain = fixed 5-class "
                     "elevation label (physical thresholds, not estimated)"),
        F1=dict(columns=F0_FEATS + F1_EXTRA,
                note="adds frozen REMA scalars (first terrain delivery, "
                     "SHA256 9953e39a...) + rebuilt alignment/status "
                     "(terrain_features_v3.csv, SHA256 853e3979...); "
                     "exploratory; no new terrain scales this round"),
        preprocessing=dict(
            rule="all preprocessing fit on the corresponding training "
                 "portion only; models may differ in internal encoding "
                 "(one-hot/standardization) with disclosure",
            f1_fill="REMA scalar missing filled with training-period median"))
    (OUT / "feature_manifest.json").write_text(json.dumps(
        feat, ensure_ascii=False, indent=2))

    # ---- 候选与 S/F 预设 ----
    cands = dict(
        T_search=dict(
            criterion="full-distribution CRPS on September validation, "
                      "computed from each candidate's own complete "
                      "predictive distribution",
            BQN=CAND_BQN, QRF=CAND_QRF, LGBM=CAND_LGBM,
            note="all candidates, failures and times recorded; final T "
                 "model refit on Mar-Aug at selected config/rounds"),
        SF_presets=dict(
            BQN=dict(degree=12, width=(64, 32), lr=1e-3, max_epochs=200,
                     patience=20, batch=256,
                     inner_selection="epochs only, via inner-val stations"),
            LGBM=dict(n_estimators=300, num_leaves=31, min_data_in_leaf=50,
                      learning_rate=0.05, subsample=0.9, subsample_freq=1,
                      colsample_bytree=0.9,
                      note="original spatial default structure, 22 quantile "
                           "targets"),
            QRF=dict(n_estimators=500, min_samples_leaf=50,
                     max_features=1.0)),
        mixture=dict(seeds=SEEDS, rule="equal-weight CDF mixture of the "
                                       "three seed models is THE official "
                                       "prediction; median/quantiles/"
                                       "probabilities/CRPS derived from "
                                       "the mixture, never averaged across "
                                       "members"))
    (OUT / "candidates.json").write_text(json.dumps(
        cands, ensure_ascii=False, indent=2, default=str))

    md = f"""# AWF_BQN_LGBM_QRF_R1 协议冻结（阶段1）

冻结时间: {datetime.now(timezone.utc).isoformat()} | 运行号 AWF_E10_PROTOCOL_R1
依据: 评审目录 10_统一执行方案 §三–§五; 台账 E01/E02。

## 数据(核验通过, data_audit.json)
57,700 有效行 / 29,216 唯一站-有效时刻 / 50 站; T 测试 18,235 行 42 站; 159 九月起报边界行保留; 强风 546 行。键 (station, init, lead) 唯一; 按有效时刻划分保证同站-有效时刻多时效不跨集。

## 三设计(splits_summary.json)
- T: 3–8月训练 / 9月内层选择 / 10–12月测试。
- S: 50 折全月 LOO; 内层验证站 = 确定性站ID md5 哈希 <20% (实际 {len(inner_val)} 站, 名单在 JSON); 不接触外层目标站。
- F: 目标站 10–12月测试; 其余站 ≤8/31 训练; 9月不用; 内层 = 哈希站 8月记录; 全 50 站可评分性保存, 主报 ≥50 测试行的站。

## 共同输入(feature_manifest.json)
F0 = 13 列(五 GFS 变量/风速/时效/月正余弦/经纬度/海拔/高程类别); 无站 ID 无站史。F1 = +6 冻结 REMA 标量 + 对齐度/状态。

## 候选与种子(candidates.json)
T 搜索: BQN 8 / QRF 8 / LGBM 8 预登记组合, 9月完整 CRPS 选择; S/F 用预设结构(内层只选轮数)。正式预测 = 种子 {SEEDS} 的等权 CDF 混合; 一切正式指标由混合分布导出。

## 资源估计(串行, 24 线程)
T: LGBM 8×3×22 分位 ≈ 1–1.5h; BQN 8×3 ≈ 0.5–1h; QRF 8×3 ≈ 0.5h。
S: 50折×3种子×3模型 ≈ 4–8h; F 同量级。F1 消融再 ×2。预计总串行 1–2 天, 分链后台执行。
"""
    (OUT / "protocol.md").write_text(md, encoding="utf-8")
    print("[protocol] frozen ->", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
