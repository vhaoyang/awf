#!/usr/bin/env python
"""Fit nonhomogeneous Gaussian regression baselines by forecast lead.

Wind-speed and u/v comparisons are evaluated on their respective eligible
rows. The Gaussian wind-speed distribution is not truncated at zero.
Run ID: AWF_E5I_EMOS_R1."""
import os
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

DATA = Path(os.environ.get("AWF_DATA_ROOT", "data"))
PAIR = DATA / "derived" / "paired" / "paired_2021.parquet"
OUT = Path("outputs/e5_baseline/emos")


def crps_normal(y, m, s):
    z = (y - m) / s
    return s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def fit_emos(x, y):
    def obj(p):
        a, b, c, d = p
        if c <= 0:
            return 1e9
        m = a + b * x
        s = c + d * x
        s = np.maximum(s, 1e-3)
        return float(np.mean(crps_normal(y, m, s)))

    best, best_v = None, np.inf
    sy = y.std()
    for m0 in ([y.mean(), 1.0, sy, 0.0],
               [y.mean(), 0.9, 0.5 * sy, 0.05],
               [0.0, 1.0, sy, 0.0]):
        r = minimize(obj, m0, method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-4, fatol=1e-5))
        if r.fun < best_v:
            best_v, best = r.fun, r.x
    return best


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    run_id = "AWF_E5I_EMOS_R1"
    df = pd.read_parquet(PAIR).rename(columns={"t2m_x": "obs_t2m",
                                               "t2m_y": "t2m"})
    df["fcst_spd"] = np.hypot(df.u10, df.v10)
    df["month"] = df.valid_time.dt.month
    df = df.dropna(subset=["wspd"]).reset_index(drop=True)
    tr = df[df.month <= 8]
    te = df[df.month >= 10].reset_index(drop=True)
    y = te.wspd.values
    print(f"[{run_id}] train={len(tr)} test={len(te)}")

    # --- wspd EMOS 逐 lead ---
    q10 = np.full(len(te), np.nan)
    q50 = np.full(len(te), np.nan)
    q90 = np.full(len(te), np.nan)
    pars = {}
    for lh in sorted(te.lead_h.unique()):
        trl = tr[tr.lead_h == lh]
        p = fit_emos(trl.fcst_spd.values, trl.wspd.values)
        pars[int(lh)] = [round(float(v), 4) for v in p]
        m_ = te[te.lead_h == lh]
        i = m_.index.values
        mu = p[0] + p[1] * m_.fcst_spd.values
        sd = np.maximum(p[2] + p[3] * m_.fcst_spd.values, 1e-3)
        q10[i] = mu + norm.ppf(0.1) * sd
        q50[i] = mu
        q90[i] = mu + norm.ppf(0.9) * sd
        print(f"  lead {lh}: a={p[0]:.2f} b={p[1]:.3f} c={p[2]:.2f} "
              f"d={p[3]:.4f}")

    pin = {}
    for tau, q in [(0.1, q10), (0.5, q50), (0.9, q90)]:
        z = y - q
        pin[tau] = float(np.mean(np.maximum(tau * z, (tau - 1) * z)))
    row_spd = dict(
        method="EMOS", n=len(te),
        spd_mae=float(np.abs(q50 - y).mean()),
        spd_rmse=float(np.sqrt(((q50 - y) ** 2).mean())),
        spd_bias=float((q50 - y).mean()),
        pinball_0_1=round(pin[0.1], 4), pinball_0_5=round(pin[0.5], 4),
        pinball_0_9=round(pin[0.9], 4),
        crps3q=float(2 * np.mean([pin[0.1], pin[0.5], pin[0.9]])),
    )
    tail = y > 15.0
    row_spd["tail_bias"] = float((q50[tail] - y[tail]).mean())
    row_spd["tail_pod_q90"] = float((q90[tail] > 15).mean())
    row_spd["tail_far_q90"] = float((q90[~tail] > 15).mean())

    # --- u/v EMOS (方向对照) ---
    def emos_uv(fcol, ocol):
        pred = np.full(len(te), np.nan)
        for lh in sorted(te.lead_h.unique()):
            trl = tr[(tr.lead_h == lh) & tr[ocol].notna()]
            p = fit_emos(trl[fcol].values, trl[ocol].values)
            m_ = te[te.lead_h == lh]
            pred[m_.index.values] = p[0] + p[1] * m_[fcol].values
        return pred

    eu = emos_uv("u10", "obs_u")
    ev = emos_uv("v10", "obs_v")
    reg = pd.read_csv(DATA / "derived/station_qc/station_sensor_registry.csv")
    m = te.dir_trust.values & te.obs_u.notna().values & ~np.isnan(eu) \
        & ~np.isnan(ev)
    derr = (np.degrees(np.arctan2(-eu[m], -ev[m])) - te.wdir.values[m]
            + 180) % 360 - 180
    row_spd["dir_mae"] = float(np.abs(derr).mean())
    row_spd["dir_n"] = int(m.sum())

    pd.DataFrame([row_spd]).round(3).to_csv(OUT / "metrics_emos.csv",
                                            index=False)
    summary = dict(
        run_id=run_id, utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        params_by_lead=pars, metrics={k: (round(v, 3) if isinstance(v, float)
                                          else v)
                                      for k, v in row_spd.items()},
        denominators="测试行(月>=10); 方向=dir_trust 行; 尾部=obs>15 行",
    )
    (OUT / "emos_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
