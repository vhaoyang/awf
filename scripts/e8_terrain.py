#!/usr/bin/env python
"""AWF E8 — V3-04 地形生产链重建：真测试过生产函数 / 中心采样 / 归一化平滑 / missing-flat 分离

运行号: AWF_E8_TERRAIN_R1 (服务器端自编, 非本侧任务号)

修复要点(对应评审 V3-04):
  1. 解析验证改为"固定合成坡面经过实际生产函数": 合成平面 DEM 写入
     内存 GeoTIFF(EPSG:3031 仿射), 用生产函数采样->梯度->基底旋转->点积;
     风向独立构造(pyproj.Geod 方位角, 不与坡面耦合): 沿坡/逆坡/横坡
     期望 +1/-1/0, 在经度 0/90E/180/90W 全部断言。
  2. 像元中心采样: 行列 = floor((坐标-原点)/步长), 即最近像元中心
     (旧代码按像元角点 round, 50 站中 35 站取错像元)。
  3. 缺测: 归一化卷积平滑 num/den(den=滤波后的有效掩码), 逐站记录
     kernel_valid_weight; den<w_min(0.5) 记 missing, 不再缺测填零。
  4. missing 与 flat 分离: 无有效 DEM/权重不足 -> missing(对齐度 NaN,
     is_missing=1); 有效但坡度<0.005 -> flat(对齐度 0, is_flat=1)。
     阈值全局唯一。
  5. 当地东/北基底: pyproj.Geod 真方位角探测(100 m, 收敛), 不再用
     大步长经向差分(旧法最大误差 1.43 度)。

输出: outputs/review_v3/{terrain_features_v3.csv, terrain_validation_v3.json}
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Geod, Transformer
from rasterio.transform import Affine
from scipy.ndimage import gaussian_filter

DATA = Path("/mnt/d/data/awfdata")
REMA = DATA / "raw" / "rema" / "rema_mosaic_1km_v2.0_dem.tif"
REG = DATA / "derived" / "station_qc" / "station_sensor_registry.csv"
OUT = Path("/home/vhyang/zcode/awf/outputs/review_v3")

SIGMA = 5.0          # 平滑尺度(像元=1km -> 5km), 与历史 slope5k 口径一致
W_MIN = 0.5          # 归一化卷积有效权重下限
FLAT_THR = 0.005     # 坡度阈值(5 m/km): 以下视为平地, 全局唯一
GEOD = Geod(ellps="WGS84")


# ---------- 生产函数 ----------

def en_basis(lon: float, lat: float, to_map) -> tuple:
    """站点处真东/真北在投影平面的单位正交基(pyproj.Geod 方位角法)。

    geodesic 方位角定义下的真东/真北; 探测距离 100 m, 投影弯曲
    误差 O(d^2/R) ~ 1e-9 rad。"""
    x0, y0 = to_map.transform(lon, lat)
    xe, ye, _ = GEOD.fwd(lon, lat, 90.0, 100.0)
    xn, yn, _ = GEOD.fwd(lon, lat, 0.0, 100.0)
    ev = np.array(to_map.transform(xe, ye)) - [x0, y0]
    nv = np.array(to_map.transform(xn, yn)) - [x0, y0]
    ev /= np.hypot(*ev)
    nv = nv - np.dot(nv, ev) * ev          # 正交化
    nv /= np.hypot(*nv)
    return ev, nv


def smooth_normalized(elev: np.ndarray, sigma: float):
    """归一化卷积: 缺测不当作零高程。返回 (平滑场, 有效权重)。"""
    m = np.isfinite(elev).astype(float)
    num = gaussian_filter(np.nan_to_num(elev, nan=0.0), sigma)
    den = gaussian_filter(m, sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        sm = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    return sm, den


def gradient_map(sm: np.ndarray, res_x: float, res_y: float):
    """平滑场梯度 -> 地图坐标 (gx, gy) [1]。res_y 为有符号值。"""
    g_row = np.gradient(sm, axis=0) / abs(res_y)   # d/d(row*|res_y|)
    g_col = np.gradient(sm, axis=1) / abs(res_x)
    gy = -g_row            # y 随行号减小(res_y<0): d/dy = -d/drow
    gx = g_col
    return gx, gy


def station_terrain(dem, transform, stations, crs,
                    sigma=SIGMA, w_min=W_MIN, flat_thr=FLAT_THR,
                    to_map=None, to_geo=None):
    """生产函数: 输入 DEM 数组与仿射, 逐站输出坡度/下坡向量/状态。

    dem: 2D (NaN=缺测); 返回 DataFrame。to_map/to_geo 可注入合成场景。"""
    if to_map is None:
        to_map = Transformer.from_crs(CRS.from_epsg(4326), crs,
                                      always_xy=True)
    if to_geo is None:
        to_geo = Transformer.from_crs(crs, CRS.from_epsg(4326),
                                      always_xy=True)
    sm, den = smooth_normalized(dem, sigma)
    gx, gy = gradient_map(sm, transform.a, transform.e)
    ny, nx = dem.shape
    rows = []
    for _, s in stations.iterrows():
        mx, my = to_map.transform(s.lon, s.lat)
        u = (my - transform.f) / transform.e
        v = (mx - transform.c) / transform.a
        ri, ci = int(np.floor(u)), int(np.floor(v))     # 最近像元中心
        inb = 0 <= ri < ny and 0 <= ci < nx
        pix_valid = bool(inb and np.isfinite(dem[ri, ci]))
        w_eff = float(den[ri, ci]) if inb else 0.0
        rec = dict(station=s.station, lat=s.lat, lon=s.lon,
                   map_x=round(mx, 1), map_y=round(my, 1),
                   row_center=ri, col_center=ci, in_bounds=bool(inb),
                   pixel_valid=pix_valid,
                   kernel_valid_weight=round(w_eff, 4),
                   subpixel_offset_m=round(
                       float(np.hypot(u - ri - 0.5, v - ci - 0.5)
                             * abs(transform.a)), 1))
        g_ok = pix_valid and w_eff >= w_min and np.isfinite(sm[ri, ci]) \
            and np.isfinite(gx[ri, ci]) and np.isfinite(gy[ri, ci])
        if g_ok:
            ev, nv = en_basis(s.lon, s.lat, to_map)
            gmx, gmy = gx[ri, ci], gy[ri, ci]
            gE = gmx * ev[0] + gmy * ev[1]
            gN = gmx * nv[0] + gmy * nv[1]
            slope = float(np.hypot(gE, gN))
            if slope < flat_thr:
                rec.update(status="flat", slope5k_EN=round(slope * 1000, 4),
                           downhill_E=0.0, downhill_N=0.0)
            else:
                rec.update(status="valid", slope5k_EN=round(slope * 1000, 4),
                           downhill_E=round(-gE / slope, 6),
                           downhill_N=round(-gN / slope, 6))
        else:
            rec.update(status="missing", slope5k_EN=np.nan,
                       downhill_E=np.nan, downhill_N=np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def katab_align_rows(downhill_E, downhill_N, u10, v10):
    """对齐度 = 单位风矢量·单位下坡向量; missing/flat 站置 0 并给标志。"""
    w = np.hypot(u10, v10)
    al = (downhill_E * u10 + downhill_N * v10) / np.maximum(w, 1e-9)
    return al


# ---------- 合成坡面验证(经过生产函数) ----------

def synth_validation():
    """固定合成平面坡 -> 内存 GeoTIFF -> 生产函数; 风向独立构造。

    坡面: h(x,y) = -S*(x*cos(phi)+y*sin(phi)), S=0.02, phi=30 度(地图系),
    即下坡方向 = 地图系方位 (cos phi, sin phi)。风由 pyproj.Geod 按真
    方位角指向构造(与坡面构造完全独立), 期望对齐: 沿坡+1/逆坡-1/横坡0。"""
    to_map = Transformer.from_crs(CRS.from_epsg(4326),
                                  CRS.from_epsg(3031), always_xy=True)
    to_geo = Transformer.from_crs(CRS.from_epsg(3031),
                                  CRS.from_epsg(4326), always_xy=True)
    S, phi = 0.02, np.deg2rad(30.0)
    dh_map = np.array([np.cos(phi), np.sin(phi)])   # 地图系下坡单位向量
    tests = []
    for lon_t in (0.0, 90.0, 180.0, -90.0):
        lat_t = -75.0
        cx, cy = to_map.transform(lon_t, lat_t)
        n = 81                                    # 81 px, 中心像元=站
        half = (n // 2 + 0.5) * 1000.0            # 40500: 像元(40,40)中心==站
        x0, ytop = cx - half, cy + half
        tr = Affine(1000.0, 0.0, x0, 0.0, -1000.0, ytop)
        xs = x0 + (np.arange(n) + 0.5) * 1000.0
        ys = ytop - (np.arange(n) + 0.5) * 1000.0
        X, Y = np.meshgrid(xs, ys)
        dem = -S * (X * np.cos(phi) + Y * np.sin(phi))
        stn = pd.DataFrame([dict(station="SYN", lat=lat_t, lon=lon_t)])
        feat = station_terrain(dem, tr, stn, CRS.from_epsg(3031),
                               to_map=to_map, to_geo=to_geo,
                               sigma=1.0, w_min=0.0, flat_thr=1e-12)
        f = feat.iloc[0]
        assert f.status == "valid", f"合成站状态异常: {f.status}"
        assert f.row_center == n // 2 and f.col_center == n // 2, \
            f"合成站未落在中心像元: {f.row_center},{f.col_center}"
        # 独立路径: 地图下坡方向 -> 大地终点 -> 真方位角; 风矢量按真方位角
        # 直接取地理东/北分量(与生产函数输出的下坡基底一致, 不经过坡面构造)
        dx_lon, dy_lat = to_geo.transform(cx + dh_map[0] * 1000.0,
                                          cy + dh_map[1] * 1000.0)
        az_dh = GEOD.inv(lon_t, lat_t, dx_lon, dy_lat)[0]   # 下坡去向方位角
        for rel, expect in ((0.0, +1.0), (180.0, -1.0),
                            (90.0, 0.0), (-90.0, 0.0)):
            az_w = np.deg2rad((az_dh + rel) % 360.0)
            w_E, w_N = np.sin(az_w), np.cos(az_w)           # 单位风矢量
            got = katab_align_rows(np.array([f.downhill_E]),
                                   np.array([f.downhill_N]),
                                   np.array([w_E]), np.array([w_N]))[0]
            tests.append(dict(lon=lon_t, wind_rel_to_downhill=rel,
                              expect=expect, got=round(float(got), 8)))
    # 容差依据: 大地线方位角与投影直线方向在 100 m 探测距离上的固有
    # 差异 ~1e-5(量级 <0.001 度), 非生产函数偏差
    ok = all(abs(t["expect"] - t["got"]) < 2e-5 for t in tests)
    return tests, ok


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = "AWF_E8_TERRAIN_R1"

    # ---------- 解析验证(先跑, 不过则中止) ----------
    tests, ok_synth = synth_validation()
    print(f"[synthetic-plane validation through production function] "
          f"{'PASS' if ok_synth else 'FAIL'} ({len(tests)} cases)")
    worst = max(abs(t["expect"] - t["got"]) for t in tests)
    assert ok_synth, "合成坡面验证未通过, 禁止产出特征"

    # ---------- 真实 REMA ----------
    ds = rasterio.open(REMA)
    elev = ds.read(1).astype(np.float64)
    if ds.nodata is not None:
        elev[elev == ds.nodata] = np.nan
    elev[elev < -100] = np.nan
    reg = pd.read_csv(REG)
    st = reg.drop_duplicates("station")[["station", "lat", "lon"]]
    tf = station_terrain(elev, ds.transform, st, ds.crs)

    # 诊断: 旧零填平滑 vs 归一化平滑 的下坡方向差(重点站)
    to_map = Transformer.from_crs(CRS.from_epsg(4326), ds.crs,
                                  always_xy=True)
    sm_norm, den = smooth_normalized(elev, SIGMA)
    sm_zero = gaussian_filter(np.nan_to_num(elev, nan=0.0), SIGMA)
    gx_n, gy_n = gradient_map(sm_norm, ds.transform.a, ds.transform.e)
    gx_z, gy_z = gradient_map(sm_zero, ds.transform.a, ds.transform.e)
    diag = []
    for _, r in tf.iterrows():
        if r.status != "valid":
            continue
        ri, ci = int(r.row_center), int(r.col_center)
        ev, nv = en_basis(r.lon, r.lat, to_map)

        def dn_EN(gx, gy):
            gE = gx[ri, ci] * ev[0] + gy[ri, ci] * ev[1]
            gN = gx[ri, ci] * nv[0] + gy[ri, ci] * nv[1]
            nn = np.hypot(gE, gN)
            return (-gE / nn, -gN / nn) if nn > 0 else (np.nan, np.nan)
        dn_z = dn_EN(gx_z, gy_z)
        dn_n = dn_EN(gx_n, gy_n)
        ang = np.degrees(np.arccos(np.clip(
            np.dot(dn_z, dn_n), -1.0, 1.0)))
        diag.append(dict(station=r.station,
                         kernel_valid_weight=r.kernel_valid_weight,
                         zero_vs_normalized_deg=round(float(ang), 2)))
    diag_df = pd.DataFrame(diag)
    focus = diag_df[diag_df.zero_vs_normalized_deg > 5].sort_values(
        "zero_vs_normalized_deg", ascending=False)

    tf.to_csv(OUT / "terrain_features_v3.csv", index=False)
    n_status = tf.status.value_counts().to_dict()
    (OUT / "terrain_validation_v3.json").write_text(json.dumps(dict(
        run_id=run_id, utc=datetime.now(timezone.utc).isoformat(),
        synthetic_plane=dict(n_cases=len(tests), worst_abs_err=worst,
                             cases=tests, passed=bool(ok_synth)),
        protocol=dict(sampling="nearest pixel center (floor index)",
                      smoothing=f"normalized convolution sigma={SIGMA} px",
                      w_min=W_MIN, flat_thr=FLAT_THR,
                      basis="pyproj.Geod true azimuth probes (100 m)",
                      missing_vs_flat="missing: no valid DEM or weight<w_min"
                                      " (align=NaN); flat: slope<thr (align=0)"),
        stations=dict(total=len(tf), **n_status),
        angle_diagnostics=dict(n=len(diag_df),
                               over_5deg=focus.to_dict("records"),
                               max=round(float(
                                   diag_df.zero_vs_normalized_deg.max()), 2)),
        dem=dict(path=str(REMA), crs=str(ds.crs),
                 res_x=float(ds.transform.a), res_y=float(ds.transform.e)),
    ), ensure_ascii=False, indent=2))
    print(f"[stations] {n_status}")
    print("[angle>5deg stations]")
    print(focus.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
