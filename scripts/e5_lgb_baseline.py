#!/usr/bin/env python
"""AWF E5 — LightGBM 分位数订正基线 vs 经典基线（S3 最小可行基线）

运行号: AWF_E5_LGB_R1 (服务器端自编, 非本侧任务号)

设计（2026-10-23 与昊哥确认）:
  划分   时间块: 训练 3-8 月 / 早停验证 9 月 / 测试 10-12 月 (valid_time 口径),
         杜绝随机划分的 3h 自相关泄漏; LOO-station 放二期。
  目标   obs wspd / obs_u / obs_v 各 τ∈{0.1,0.5,0.9}; u/v 仅用 dir_trust 行
         (风速<2 m/s 风向不可信, obs_u/obs_v=NaN)。
  方法   (1) 原始GFS  (2) QM: 逐 lead 经验分位映射 F_obs^-1(F_fcst(x))
         (3) MOS: 逐 lead 线性回归 obs~fcst  (4) 持续性: init 时刻最近观测
         (5) LightGBM quantile (特征: GFS 五变量+fcst_spd+lead+月sin/cos
         +站经纬/高程/地形类)。
  指标   MAE/RMSE(spd,点), pinball(τ), CRPS 3 分位近似 (2/K)*Σpinball,
         大风尾部(obs>15 m/s): 点法偏差 + LGBM q90>15 的 POD/FAR,
         风向圆周误差(dir_trust): circ mean/MAE。
  口径   逐指标注明分母(测试行数/trust行数/尾部行数); 持续性缺早期行单独计 n。

输出: outputs/e5_baseline/{e5_summary.json, metrics_point.csv, metrics_quantile.csv,
       metrics_tail.csv, metrics_direction.csv, config.json}
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA = Path("/mnt/d/data/awfdata")
PAIR = DATA / "derived" / "paired" / "paired_2021.parquet"
REG = DATA / "derived" / "station_qc" / "station_sensor_registry.csv"
OUT = Path("/home/vhyang/zcode/awf/outputs/e5_baseline")

SEED = 42
THREADS = 24
QUANTILES = [0.1, 0.5, 0.9]
LGB_PARAMS = dict(objective="quantile", n_estimators=400, learning_rate=0.05,
                  num_leaves=31, min_data_in_leaf=50, subsample=0.9,
                  subsample_freq=1, colsample_bytree=0.9, seed=SEED,
                  num_threads=THREADS, deterministic=True, force_col_wise=True,
                  verbosity=-1)
TRAIN_END = 8   # 训练: 月 3-8; 验证: 9; 测试: 10-12
VAL_MONTH = 9
TAIL_THR = 15.0

FEATS = ["u10", "v10", "fcst_spd", "t2m", "prmsl", "rh2m", "lead_h",
         "month_sin", "month_cos", "lat", "lon", "elevation", "terrain"]


def circ_deg(d):
    return (d + 180.0) % 360.0 - 180.0


def crps_approx(pin: dict) -> float:
    """CRPS ≈ (2/K) Σ_τ pinball_τ (K 分位近似, 此处 K=3, 口径已声明)。"""
    return float(2.0 * np.mean([pin[f"pinball_{t}"] for t in QUANTILES]))


def qm_map(tr_fcst, tr_obs, x):
    """经验分位映射: x 在训练期 fcst 分布中的分位 -> 训练期 obs 同分位值。
    训练对先剔除任一侧缺失 (dir_trust 但 wdir 缺失的行 obs_u=NaN)。"""
    ok = ~np.isnan(tr_fcst) & ~np.isnan(tr_obs)
    xs, os_ = np.sort(tr_fcst[ok]), np.sort(tr_obs[ok])
    if len(xs) == 0:
        return np.full_like(x, np.nan, dtype=float)
    r = np.searchsorted(xs, x, side="right") / len(xs)
    return np.quantile(os_, np.clip(r, 0, 1))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = "AWF_E5_LGB_R1"

    df = pd.read_parquet(PAIR)
    df = df.rename(columns={"t2m_x": "obs_t2m", "t2m_y": "t2m"})
    reg = pd.read_csv(REG)
    df = df.merge(reg[["station", "lat", "lon", "elevation", "terrain"]],
                  on="station", how="left")
    df["terrain"] = df.terrain.astype("category")
    df["fcst_spd"] = np.hypot(df.u10, df.v10)
    df["month"] = df.valid_time.dt.month
    df["month_sin"] = np.sin(2 * np.pi * df.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.month / 12)

    # 行过滤: wspd 缺失行全部剔除(分母声明)
    n0 = len(df)
    df = df.dropna(subset=["wspd"]).reset_index(drop=True)
    df["init_t"] = df.valid_time - pd.to_timedelta(df.lead_h, unit="h")

    tr = df[df.month <= TRAIN_END]
    va = df[df.month == VAL_MONTH]
    te = df[df.month >= 10].reset_index(drop=True)
    print(f"[{run_id}] rows total={n0} usable={len(df)} "
          f"train={len(tr)} val={len(va)} test={len(te)} "
          f"stations={df.station.nunique()}")

    # ---------- (4) 持续性: 各站 init 时刻(含)最近一次观测 (向量化, 不做合并) ----------
    te = te.reset_index(drop=True)
    obs_sort = df.sort_values(["station", "valid_time"])
    te["pers_spd"] = np.nan; te["pers_u"] = np.nan; te["pers_v"] = np.nan
    for s, og in obs_sort.groupby("station"):
        idxs = np.where(te.station.values == s)[0]
        if len(idxs) == 0:
            continue
        pos = np.searchsorted(og.valid_time.values,
                              te.init_t.values[idxs], side="right") - 1
        ok = pos >= 0
        te.iloc[idxs[ok], te.columns.get_indexer(
            ["pers_spd", "pers_u", "pers_v"])] = np.column_stack([
                og.wspd.values[pos[ok]], og.obs_u.values[pos[ok]],
                og.obs_v.values[pos[ok]]])

    # ---------- (2) QM 逐 lead ----------
    qm_spd, qm_u, qm_v = (np.full(len(te), np.nan) for _ in range(3))
    for lh, idx in te.groupby("lead_h").groups.items():
        i = np.asarray(idx)
        trl = tr[tr.lead_h == lh]
        trt = trl[trl.dir_trust]
        qm_spd[i] = qm_map(trl.fcst_spd.values, trl.wspd.values,
                           te.fcst_spd.values[i])
        qm_u[i] = qm_map(trt.u10.values, trt.obs_u.values, te.u10.values[i])
        qm_v[i] = qm_map(trt.v10.values, trt.obs_v.values, te.v10.values[i])

    # ---------- (3) MOS 逐 lead 线性 ----------
    mos = {}
    for tgt, fc in [("wspd", "fcst_spd"), ("obs_u", "u10"), ("obs_v", "v10")]:
        pred = np.full(len(te), np.nan)
        for lh in te.lead_h.unique():
            trl = tr[(tr.lead_h == lh) & tr[tgt].notna()]
            m = (te.lead_h == lh).values
            pred[m] = LinearRegression().fit(
                trl[[fc]].values, trl[tgt].values).predict(te[[fc]].values[m])
        mos[tgt] = pred

    # ---------- (5) LightGBM 分位数 ----------
    cat = FEATS.index("terrain")
    preds = {}
    for tgt in ["wspd", "obs_u", "obs_v"]:
        trt = tr[tr[tgt].notna()]
        vat = va[va[tgt].notna()]
        for tau in QUANTILES:
            m = lgb.LGBMRegressor(**{**LGB_PARAMS, "objective": "quantile",
                                     "alpha": tau})
            m.fit(trt[FEATS], trt[tgt], categorical_feature=[cat],
                  eval_set=[(vat[FEATS], vat[tgt])],
                  eval_metric="quantile",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            preds[(tgt, tau)] = m.predict(te[FEATS],
                                          num_iteration=m.best_iteration_)
            print(f"  fitted {tgt} tau={tau} best_iter={m.best_iteration_}")

    # ================= 评估 =================
    y = te.wspd.values
    trust = te.dir_trust.values & te.obs_u.notna().values
    rows_point, rows_q, tail_rows, dir_rows = [], [], [], []

    def add_point(method, pred, mask=None):
        m = mask if mask is not None else np.ones(len(te), bool)
        m = m & ~np.isnan(pred)
        e = pred[m] - y[m]
        rows_point.append(dict(method=method, n=int(m.sum()),
                               spd_mae=float(np.abs(e).mean()),
                               spd_rmse=float(np.sqrt((e ** 2).mean())),
                               spd_bias=float(e.mean())))

    add_point("GFS_raw", te.fcst_spd.values)
    add_point("QM", qm_spd)
    add_point("MOS", mos["wspd"])
    add_point("Persistence", te.pers_spd.values)
    add_point("LGBM_q50", preds[("wspd", 0.5)])

    # 分位指标: pinball(τ,q,y)=max(τ(y-q), (τ-1)(y-q)); 点法无分布, q10/q90
    # 退化为点值(对照口径, 已声明)
    for method, q10, q50, q90 in [
            ("GFS_raw", te.fcst_spd.values, te.fcst_spd.values, te.fcst_spd.values),
            ("QM", qm_spd, qm_spd, qm_spd),
            ("LGBM", preds[("wspd", 0.1)], preds[("wspd", 0.5)],
             preds[("wspd", 0.9)])]:
        pin = {}
        for tau in QUANTILES:
            q = {0.1: q10, 0.5: q50, 0.9: q90}[tau]
            z = y - q
            pin[f"pinball_{tau}"] = float(
                np.mean(np.maximum(tau * z, (tau - 1) * z)))
        rows_q.append(dict(method=method, n=len(te), **pin,
                           crps3q=crps_approx(pin)))

    # 大风尾部 (obs > 15 m/s)
    tail = te.wspd > TAIL_THR
    for method, pred in [("GFS_raw", te.fcst_spd.values), ("QM", qm_spd),
                         ("MOS", mos["wspd"]), ("LGBM_q50",
                                                preds[("wspd", 0.5)])]:
        e = pred[tail] - y[tail]
        tail_rows.append(dict(method=method, n_tail=int(tail.sum()),
                              spd_bias=float(e.mean()),
                              spd_mae=float(np.abs(e).mean())))
    q90 = preds[("wspd", 0.9)]
    pod = float((q90[tail] > TAIL_THR).mean())
    far = float((q90[~tail] > TAIL_THR).mean())
    tail_rows.append(dict(method="LGBM_q90>15", n_tail=int(tail.sum()),
                          spd_bias=float(pod), spd_mae=float(far)))

    # 风向 (圆周, dir_trust)
    for method, u, v in [
            ("GFS_raw", te.u10.values, te.v10.values),
            ("QM", qm_u, qm_v),
            ("MOS", mos["obs_u"], mos["obs_v"]),
            ("Persistence", te.pers_u.values, te.pers_v.values),
            ("LGBM_q50", preds[("obs_u", 0.5)], preds[("obs_v", 0.5)])]:
        m = trust & ~np.isnan(u) & ~np.isnan(v)
        pdir = np.degrees(np.arctan2(-u[m], -v[m]))
        derr = circ_deg(pdir - te.wdir.values[m])
        circ_mean = float(np.degrees(np.arctan2(
            np.mean(np.sin(np.radians(derr))),
            np.mean(np.cos(np.radians(derr))))))
        dir_rows.append(dict(method=method, n=int(m.sum()),
                             dir_circ_mean_deg=round(circ_mean, 2),
                             dir_mae_deg=float(np.abs(derr).mean())))

    pd.DataFrame(rows_point).round(3).to_csv(OUT / "metrics_point.csv", index=False)
    pd.DataFrame(rows_q).round(4).to_csv(OUT / "metrics_quantile.csv", index=False)
    pd.DataFrame(tail_rows).round(3).to_csv(OUT / "metrics_tail.csv", index=False)
    pd.DataFrame(dir_rows).round(2).to_csv(OUT / "metrics_direction.csv", index=False)

    # 预测落盘 (图数据/复核用): 关键列 + 各方法预测
    pred_tbl = te[["station", "valid_time", "lead_h", "wspd", "wdir",
                   "dir_trust"]].copy()
    pred_tbl["gfs_spd"] = te.fcst_spd.values
    pred_tbl["qm_spd"] = qm_spd
    pred_tbl["mos_spd"] = mos["wspd"]
    pred_tbl["pers_spd"] = te.pers_spd.values
    for tgt, col in [("wspd", "spd"), ("obs_u", "u"), ("obs_v", "v")]:
        for tau in QUANTILES:
            pred_tbl[f"lgbm_{col}_q{int(tau*100)}"] = preds[(tgt, tau)]
    pred_tbl.to_parquet(OUT / "test_predictions.parquet", index=False)

    # 对比图: MAE 与 crps3q
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    pt = pd.DataFrame(rows_point)
    axes[0].bar(pt.method, pt.spd_mae, color="steelblue")
    axes[0].set_title("wind-speed MAE (m/s), test Oct-Dec")
    axes[0].tick_params(axis="x", rotation=20)
    qt = pd.DataFrame(rows_q)
    axes[1].bar(qt.method, qt.crps3q, color="darkorange")
    axes[1].set_title("CRPS (3-quantile approx., m/s)")
    axes[1].tick_params(axis="x", rotation=20)
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_method_comparison.png", dpi=150)
    plt.close(fig)

    summary = dict(
        run_id=run_id, utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        n_rows_total=n0, n_rows_usable=len(df),
        split=dict(train="month<=8", val="month==9", test="month>=10",
                   n_train=len(tr), n_val=len(va), n_test=len(te)),
        n_stations=int(df.station.nunique()),
        denominators=dict(
            point="测试行(月>=10, wspd 非缺失); Persistence 缺早期行 n 单计",
            quantile="全部测试行; GFS/QM 无分布, q10/q90 退化为点值(对照口径)",
            tail=f"obs>{TAIL_THR} m/s 测试行",
            direction="dir_trust(风速>=2m/s) 且 u/v 非缺失测试行"),
        versions=dict(lightgbm=lgb.__version__, pandas=pd.__version__,
                      numpy=np.__version__),
    )
    (OUT / "e5_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    (OUT / "config.json").write_text(json.dumps(
        dict(seed=SEED, threads=THREADS, quantiles=QUANTILES,
             lgbm=LGB_PARAMS, features=FEATS, train_end_month=TRAIN_END,
             val_month=VAL_MONTH, tail_thr=TAIL_THR),
        ensure_ascii=False, indent=2))
    print("== point ==");  print(pd.DataFrame(rows_point).round(3).to_string(index=False))
    print("== quantile =="); print(pd.DataFrame(rows_q).round(3).to_string(index=False))
    print("== tail =="); print(pd.DataFrame(tail_rows).round(3).to_string(index=False))
    print("== direction =="); print(pd.DataFrame(dir_rows).round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
