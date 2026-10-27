#!/usr/bin/env python
"""AWF E5J — SHAP 可解释性分析（P2r 全特征模型, wspd 分位数）

运行号: AWF_E5J_SHAP_R1 (服务器端自编, 非本侧任务号)

内容: TreeExplainer 对 P2r(13 基础特征 + 5 REMA 静态 + katab_align_rema) 的
wspd q50 模型做归因; 输出 (a) 全局重要性条形图 (b) katab_align_rema 与
fetch_drop 的依赖图 (物理可解释性证据, 供正文或补充材料取舍)。
输出: outputs/e5_baseline/shap/{fig7_shap.png, shap_importance.csv,
       shap_summary.json}
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shap
from pyproj import CRS, Transformer
from scipy.ndimage import gaussian_filter

DATA = Path("/mnt/d/data/awfdata")
PAIR = DATA / "derived" / "paired" / "paired_2021.parquet"
REG = DATA / "derived" / "station_qc" / "station_sensor_registry.csv"
REMA = DATA / "raw" / "rema" / "rema_mosaic_1km_v2.0_dem.tif"
OUT = Path("/home/vhyang/zcode/awf/outputs/e5_baseline/shap")

FEATS = ["u10", "v10", "fcst_spd", "t2m", "prmsl", "rh2m", "lead_h",
         "month_sin", "month_cos", "lat", "lon", "elevation", "terrain",
         "slope1k", "slope5k", "slope25k", "prominence", "fetch_drop",
         "katab_align_rema"]


def katab_series(df, st):
    ds = rasterio.open(REMA)
    elev = ds.read(1).astype(np.float32)
    nod = ds.nodata
    if nod is not None:
        elev[elev == nod] = np.nan
    elev[elev < -100] = np.nan
    tr, crs, res = ds.transform, ds.crs, float(np.abs(ds.transform.a))
    sm5 = gaussian_filter(elev, 5000.0 / res)
    gy, gx = np.gradient(sm5, res, res)
    to_map = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    xs, ys = to_map.transform(st.lon.values, st.lat.values)
    cx = np.clip(np.round((xs - tr.c) / tr.a).astype(int), 0, elev.shape[1] - 1)
    cy = np.clip(np.round((ys - tr.f) / tr.e).astype(int), 0, elev.shape[0] - 1)
    du, dv = -gx[cy, cx], -gy[cy, cx]
    m = dict(zip(st.station, zip(du, dv)))
    du_r = df.station.map(lambda s: m[s][0]).values
    dv_r = df.station.map(lambda s: m[s][1]).values
    vdot = du_r * df.u10.values + dv_r * df.v10.values
    norm = np.hypot(du_r, dv_r) * np.maximum(
        np.hypot(df.u10, df.v10).values, 1e-6)
    return vdot / np.maximum(norm, 1e-6)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = "AWF_E5J_SHAP_R1"
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
    df = df.dropna(subset=["wspd"]).reset_index(drop=True)
    feats = pd.read_csv("/home/vhyang/zcode/awf/outputs/e5_baseline/rema/"
                        "rema_features_by_station.csv")
    df = df.merge(feats, on="station", how="left")
    st = reg.drop_duplicates("station")[["station", "lat", "lon"]]
    df["katab_align_rema"] = katab_series(df, st)

    trn = df[df.month <= 8]
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=0.5, n_estimators=400,
        learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.9, seed=42,
        num_threads=24, deterministic=True, force_col_wise=True, verbosity=-1)
    model.fit(trn[FEATS], trn.wspd, categorical_feature=[FEATS.index("terrain")])

    samp = df[df.month >= 10].sample(n=min(3000, (df.month >= 10).sum()),
                                     random_state=42)
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(samp[FEATS])
    if isinstance(sv, list):
        sv = sv[0]

    imp = pd.DataFrame(dict(feature=FEATS,
                            mean_abs_shap=np.abs(sv).mean(axis=0))
                       ).sort_values("mean_abs_shap", ascending=False)
    imp.round(4).to_csv(OUT / "shap_importance.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8),
                             gridspec_kw={"width_ratios": [1.1, 1]})
    top = imp.head(10).iloc[::-1]
    axes[0].barh(top.feature, top.mean_abs_shap, color="#0173B2")
    axes[0].set_xlabel("mean |SHAP value| (m/s)")
    axes[0].set_title("(a) global feature importance (top 10)")
    i = FEATS.index("katab_align_rema")
    sc = axes[1].scatter(samp.katab_align_rema, sv[:, i],
                         c=samp.fetch_drop, cmap="viridis", s=6, alpha=0.6)
    axes[1].axhline(0, color="k", lw=0.7)
    axes[1].set_xlabel("katabatic alignment (cos)")
    axes[1].set_ylabel("SHAP of katabatic alignment")
    axes[1].set_title("(b) katabatic-alignment effect\n(color: fetch drop, m)")
    fig.colorbar(sc, ax=axes[1], pad=0.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_shap.png", dpi=300)
    plt.close(fig)

    # 物理一致性核验: 对齐度>0.8 vs <-0.8 的平均 SHAP
    hi = sv[samp.katab_align_rema.values > 0.8, i].mean()
    lo = sv[samp.katab_align_rema.values < -0.8, i].mean()
    summary = dict(
        run_id=run_id, utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        n_explained=len(samp),
        top5=imp.head(5).round(4).to_dict("records"),
        katab_shap_mean_align_gt08=round(float(hi), 3),
        katab_shap_mean_align_lt_m08=round(float(lo), 3),
        note="SHAP>0 表示推高风速预测; 若强下坡对齐(>0.8)的平均归因为正、"
             "逆坡(<-0.8)为负, 则模型学到了 katabatic 增强的物理方向",
    )
    (OUT / "shap_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
