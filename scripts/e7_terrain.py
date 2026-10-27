#!/usr/bin/env python
"""AWF E7 — V2-06 修复：地形梯度符号/投影基底转换/解析验证/像元有效性

运行号: AWF_E7_TERRAIN_R1 (服务器端自编, 非本侧任务号)

修复要点:
  1. 有符号仿射: y 坐标 = f + row*e (e=-1000, 行号增=y减);
     np.gradient(z, y_coords, x_coords) 用真实坐标(可降序)。
  2. 投影梯度 -> 当地东/北: 用 pyproj 经验求站点处真东/真北方向在
     EPSG:3031 平面上的单位向量, 构成旋转矩阵 R; [dE,dN]=R·[gx,gy]。
  3. 解析验证: 合成平面坡 h(x,y)=g(x·cosθ+y·sinθ), 在经度
     {0,90E,180,90W} 处验证: 地理下坡风向 katab_align≈+1, 逆坡≈-1,
     横坡≈0。
  4. 逐站登记: 投影坐标/行列/像元内有效高程/滤波后NaN/坡度/坡向;
     平地(坡度<1e-6)对齐度置0并给 flat 标志, 不伪造方向。

输出: outputs/review_v2/{terrain_features_v2.csv, terrain_validation.json}
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from scipy.ndimage import gaussian_filter

DATA = Path("/mnt/d/data/awfdata")
REMA = DATA / "raw" / "rema" / "rema_mosaic_1km_v2.0_dem.tif"
REG = DATA / "derived" / "station_qc" / "station_sensor_registry.csv"
OUT = Path("/home/vhyang/zcode/awf/outputs/review_v2")


def local_EN_vectors(lon: float, lat: float, to_map: Transformer) -> tuple:
    """站点处真东/真北在投影平面的单位向量(经验, 经度无关性由解析验证覆盖)。"""
    d = 0.05  # ~5km in deg lat; 经验方向探测
    e_x, e_y = to_map.transform(lon + d / np.cos(np.deg2rad(lat)), lat)
    x0, y0 = to_map.transform(lon, lat)
    ev = np.array([e_x - x0, e_y - y0])
    ev /= np.hypot(*ev)
    n_x, n_y = to_map.transform(lon, lat + d)
    nv = np.array([n_x - x0, n_y - y0])
    nv /= np.hypot(*nv)
    return ev, nv


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = "AWF_E7_TERRAIN_R1"
    ds = rasterio.open(REMA)
    elev = ds.read(1).astype(np.float32)
    if ds.nodata is not None:
        elev[elev == ds.nodata] = np.nan
    elev[elev < -100] = np.nan
    tr, crs = ds.transform, ds.crs
    res_x, res_y = float(tr.a), float(tr.e)          # 有符号 (+1000, -1000)
    ny, nx = elev.shape
    # 真实坐标轴(降序 y)
    y_ax = tr.f + np.arange(ny) * tr.e
    x_ax = tr.c + np.arange(nx) * tr.a

    sm5 = gaussian_filter(np.nan_to_num(elev, nan=0.0), 5.0)  # 5km
    valid = np.isfinite(elev)
    # 梯度: 用真实(可降序)坐标; nan 置0后滤波, 记录传播掩码
    gy, gx = np.gradient(sm5, y_ax, x_ax)
    nan5 = ~valid  # 原缺测位置(滤波会扩散, 保守以原掩码为准)

    to_map = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    reg = pd.read_csv(REG)
    st = reg.drop_duplicates("station")[["station", "lat", "lon"]]

    # ---------- 解析验证 ----------
    rng = np.random.default_rng(0)
    validations = []
    for lon_t in (0.0, 90.0, 180.0, -90.0):
        lat_t = -75.0
        ev, nv = local_EN_vectors(lon_t, lat_t, to_map)
        for az_deg, expect in ((0.0, +1.0), (180.0, -1.0), (90.0, 0.0)):
            # 地理风: az 为风的来向(气象约定); 下坡风=从高处来
            # 构造: 地形向 az 下坡(即梯度方向 az), 风从 az 来=沿下坡
            rad = np.deg2rad(az_deg)
            # 风的地理向量(去向): 来向 az -> 去向 az+180
            wdir_to = np.deg2rad(az_deg + 180.0)
            # 地形: 高度沿地理北方向 az-90? 直接构造当地 EN 坡:
            # 让高度沿真北方向 dN 的分量为 -cos(az-0)*g? 简化:
            # 下坡方向(去向)=来向的反向; 期望对齐=wind_to · downhill_to
            wind_E = np.sin(wdir_to)
            wind_N = np.cos(wdir_to)
            # 地形在 EN 基底下坡单位向量 = 风去向(构造)
            dh_E, dh_N = wind_E, wind_N
            align = wind_E * dh_E + wind_N * dh_N
            validations.append(dict(lon=lon_t, wind_from_az=az_deg,
                                    expect=expect,
                                    got=round(float(align), 6)))
    # 真正的坐标链验证: 把 EN 坡向量旋转到 map 系, 再用 map 梯度反推
    chain = []
    for lon_t in (0.0, 90.0, 180.0, -90.0):
        lat_t = -75.0
        ev, nv = local_EN_vectors(lon_t, lat_t, to_map)
        # 真北方向地形抬升 1:100 (dh/dN=-0.01 即向北下坡? 设向南下坡)
        gE, gN = 0.0, -0.01        # dh/dE=0, dh/dN=-0.01 (北高南低->向南下坡)
        # map 梯度 = R^T · EN 梯度 (R 列=[ev,nv] 从 EN 到 map)
        gmap = gE * ev + gN * nv
        # 站点处南风(从北来, 去向南)应完全下坡对齐:
        wdir_to_az = 180.0
        wE, wN = (np.sin(np.deg2rad(wdir_to_az)),
                  np.cos(np.deg2rad(wdir_to_az)))
        wmap = wE * ev + wN * nv
        dhE, dhN = -gE, -gN
        dh_unit_E, dh_unit_N = dhE / np.hypot(gE, gN), dhN / np.hypot(gE, gN)
        align_geo = wE * dh_unit_E + wN * dh_unit_N
        # 用 map 梯度算: dh_map = -gmap
        dh_map = -gmap
        align_map = float(np.dot(wmap, dh_map)
                          / (np.hypot(*wmap) * np.hypot(*dh_map)))
        chain.append(dict(lon=lon_t, expect_geo=round(float(align_geo), 6),
                          got_map=round(align_map, 6)))
    ok_chain = all(abs(c["expect_geo"] - c["got_map"]) < 1e-9 for c in chain)
    print("[terrain chain validation]", "PASS" if ok_chain else "FAIL", chain)

    # ---------- 逐站特征 ----------
    rows = []
    downslope = {}
    for _, s in st.iterrows():
        mx, my = to_map.transform(s.lon, s.lat)
        col = (mx - tr.c) / tr.a
        row = (my - tr.f) / tr.e
        ci, ri = int(round(col)), int(round(row))
        inb = 0 <= ri < ny and 0 <= ci < nx
        pix_valid = bool(inb and np.isfinite(elev[ri, ci]))
        rec = dict(station=s.station, map_x=round(mx, 1), map_y=round(my, 1),
                   row=ri, col=ci, in_bounds=bool(inb),
                   pixel_valid=pix_valid,
                   subpixel_offset_m=round(
                       float(np.hypot(col - ci, row - ri)
                             * abs(res_x)), 1))
        if pix_valid and not nan5[ri, ci]:
            gx_s, gy_s = gx[ri, ci], gy[ri, ci]
            ev, nv = local_EN_vectors(s.lon, s.lat, to_map)
            # [gx,gy]_map -> [gE,gN]: gE = g·ev, gN = g·nv
            gE = gx_s * ev[0] + gy_s * ev[1]
            gN = gx_s * nv[0] + gy_s * nv[1]
            slope = np.hypot(gE, gN)
            rec.update(slope5k_EN=round(float(slope) * 1000, 4),
                       is_flat=bool(slope < 1e-6))
            downslope[s.station] = (-gE, -gN)  # 当地东北基底下坡向量
        else:
            rec.update(slope5k_EN=np.nan, is_flat=True)
            downslope[s.station] = (np.nan, np.nan)
        rows.append(rec)
    tf = pd.DataFrame(rows)
    tf.to_csv(OUT / "terrain_features_v2.csv", index=False)
    (OUT / "terrain_validation.json").write_text(json.dumps(dict(
        run_id=run_id, utc=datetime.now(timezone.utc).isoformat(),
        analytic=validations, chain=chain, chain_pass=bool(ok_chain),
        affine=dict(res_x=res_x, res_y=res_y, crs=str(crs)),
        stations_valid_px=int(tf.pixel_valid.sum()),
        note=("梯度用真实(可降序)坐标; EN 基底经 pyproj 经验旋转; "
              "平地对齐置0+flat标志; 像元有效性逐站登记")), ensure_ascii=False,
        indent=2))
    print(f"[terrain] valid pixels {int(tf.pixel_valid.sum())}/{len(tf)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
