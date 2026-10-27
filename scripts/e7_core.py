#!/usr/bin/env python
"""AWF E7 — V2 返修核心库：冻结样本、合法 CDF、双协议评分、正确块 bootstrap

运行号: AWF_E7_CORE_R1 (服务器端自编, 非本侧任务号)

对应 V2-01/02/03/05:
  - 样本冻结: 57,700 有效行; 唯一 station-valid_time=29,216; 年50站/测试42站
  - 单调化: 固定升序 tau, 分位值按列排列后逐行单调化(sort), CDF 合法
  - 评分双协议: QS22(实际tau等权, 命名明确) + 完整CRPS(梯形积分0..1,
    线性插值 Q(tau), 两端尾部: tau<min 线性外推至(0, max(q-q_step,0)),
    tau>max 外推至(1, q+q_step); q_step=相邻分位间距中位数) —— 数值验证:
    Gaussian 闭式 vs 数值一致; 退化分布 CRPS==MAE
  - 块 bootstrap: 日期块有放回抽样, 重复块保留重数(索引拼接含重复),
    抽满 N_dates 个日期为止; QS3 差 = 2*mean(3 pinball 差)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------- 单调化与CDF ----------

def monotonize_rows(Q: np.ndarray) -> np.ndarray:
    """逐行单调化(非降)。Q: n x K, 列已按升序 tau 排好。"""
    return np.sort(Q, axis=1)


def cdf_at(Q_mono: np.ndarray, taus: np.ndarray, x: float) -> np.ndarray:
    """合法 P(X<=x), 逐行分段线性。两端尾部规则(文档化):
      x<=0        -> 0 (非负支撑);
      0<x<q_min   -> taus[0]*(x/q_min)          (线性至原点);
      x>q_max     -> taus[-1]+(1-taus[-1])*min((x-q_max)/(3*dq),1)
                     其中 dq=分位间距均值 (3dq 内线性到 1);
      其余        -> np.interp(x, q, taus)。
    保证 [0,1] 且随 x 不减。"""
    taus = np.asarray(taus, dtype=float)
    n = Q_mono.shape[0]
    p = np.empty(n)
    K = Q_mono.shape[1]
    for i in range(n):
        q = Q_mono[i]
        if x <= 0.0:
            p[i] = 0.0
        elif x < q[0]:
            p[i] = taus[0] * (x / max(q[0], 1e-6))
        elif x > q[-1]:
            dq = (q[-1] - q[0]) / max(K - 1, 1)
            p[i] = taus[-1] + (1 - taus[-1]) * min(
                (x - q[-1]) / max(3.0 * dq, 1e-6), 1.0)
        else:
            p[i] = np.interp(x, q, taus)
    return np.clip(p, 0.0, 1.0)


def survival_at(Q_mono, taus, x):
    return 1.0 - cdf_at(Q_mono, taus, x)


# ---------- 双协议评分 ----------

def pinball(y, q, tau):
    z = y - q
    return np.maximum(tau * z, (tau - 1) * z)


def qs_k(y, Q, taus):
    """有限分位等权得分: 2/K * sum pinball (K=len(taus), 真实 K 自动)。"""
    K = len(taus)
    return float(2.0 / K * np.mean(
        [pinball(y, Q[:, j], taus[j]).mean() for j in range(K)]))


def qs_k_rowwise(y, Q, taus):
    """行级 2/K*sum pinball (供 bootstrap)。"""
    K = len(taus)
    cols = [pinball(y, Q[:, j], taus[j]) for j in range(K)]
    return 2.0 / K * np.sum(cols, axis=0)


def crps_full(y, Q_mono, taus):
    """完整 CRPS: 2*int_0^1 pinball dtau, 分位函数线性插值+两端线性外推。
    外推斜率取相邻分位间距中位数 dq: Q(0)=max(Q[:,0]-dq,0), Q(1)=Q[:,-1]+dq。
    返回 (均值, 行级数组)。梯形积分节点 = 外推端点+全部 taus。"""
    n, K = Q_mono.shape
    dq = np.median(np.diff(Q_mono, axis=1), axis=1) if K > 1 else \
        np.full(n, 0.1)
    q0 = np.maximum(Q_mono[:, 0] - dq, 0.0)
    q1 = Q_mono[:, -1] + dq
    # 网格: tau=0 与 tau=1 端点 + 内部 taus
    grid_t = np.concatenate([[0.0], taus, [1.0]])
    vals = np.empty((n, len(grid_t)))
    vals[:, 0] = q0
    vals[:, 1:-1] = Q_mono
    vals[:, -1] = q1
    # pinball(tau) 于网格: 逐节点计算
    pb = np.empty_like(vals)
    for j, t in enumerate(grid_t):
        pb[:, j] = np.maximum(t * (y - vals[:, j]),
                              (t - 1) * (y - vals[:, j]))
    row = 2.0 * np.trapezoid(pb, grid_t, axis=1)
    return float(row.mean()), row


def crps_normal_closed(y, mu, sd):
    z = (y - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z)
                 - 1 / np.sqrt(np.pi))


# ---------- 块 bootstrap (V2-03 修复) ----------

def block_bootstrap(diff_rows: np.ndarray, dates: np.ndarray,
                    block_days: int, n_boot: int = 1000, seed: int = 42):
    """按日期块有放回重抽样, 保留重复块重数。
    dates: 逐行起报日期(可重复=同日多行)。抽样单元=连续 block_days 个
    日期窗口(按唯一日期排序); 每次重抽抽 ceil(n_dates) 个块(有放回),
    行索引按块内日期的全部行展开(重复块→行出现多次, 保留重数)。
    返回 bootstrap 均值数组。"""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(pd.unique(dates)))
    nd = len(uniq)
    n_blocks_needed = int(np.ceil(nd / block_days))
    rows_by_date = {d: np.where(dates == d)[0] for d in uniq}
    stats = np.empty(n_boot)
    for b in range(n_boot):
        block_starts = rng.integers(0, max(nd - block_days + 1, 1),
                                    size=n_blocks_needed)
        idx_parts = []
        for s in block_starts:
            for d in uniq[s:s + block_days]:
                idx_parts.append(rows_by_date[d])
        idx = np.concatenate(idx_parts)      # 含重复(有放回重数保留)
        stats[b] = diff_rows[idx].mean()
    return stats


# ---------- 验证套件 ----------

def run_selftests():
    ok = []
    # 1) Gaussian 闭式 vs 数值完整CRPS
    rng = np.random.default_rng(0)
    mu, sd = 5.0, 2.0
    y = norm.rvs(mu, sd, size=4000, random_state=1)
    taus = np.concatenate([np.arange(0.05, 0.96, 0.05),
                           [0.97, 0.98, 0.99]])
    Q = np.tile(np.array([[mu + norm.ppf(t) * sd for t in taus]]),
                (len(y), 1))
    m_num, _ = crps_full(y, Q, taus)
    m_closed = float(np.mean(crps_normal_closed(y, np.full_like(y, mu),
                                                np.full_like(y, sd))))
    ok.append(("gaussian closed vs numeric",
               abs(m_num - m_closed) < 0.02, round(m_num, 4),
               round(m_closed, 4)))
    # 2) 确定性预报 c 的完整 CRPS == MAE(c, y)
    yd = rng.normal(5, 2, 2000)
    c = 4.6                                # 任意点预报
    Qd = np.full((len(yd), len(taus)), c)  # 全分位=c(退化)
    m_deg, _ = crps_full(yd, Qd, taus)
    mae_c = float(np.abs(c - yd).mean())
    ok.append(("degenerate CRPS == MAE(point)",
               abs(m_deg - mae_c) < 0.02, round(m_deg, 4), round(mae_c, 4)))
    # 3) bootstrap 重复重数验证: 同分布差应为~0; 抽中两次的块贡献两次
    d = np.concatenate([np.zeros(100), np.ones(100)])
    dts = np.array(["2021-10-01"] * 100 + ["2021-10-02"] * 100)
    st = block_bootstrap(d, dts, block_days=1, n_boot=200, seed=3)
    ok.append(("bootstrap mean ~0.5", abs(st.mean() - 0.5) < 0.05,
               round(st.mean(), 3)))
    # 4) CDF 合法性
    Qm = monotonize_rows(Q)
    p1 = cdf_at(Qm, taus, -1.0)
    p2 = cdf_at(Qm, taus, 100.0)
    ok.append(("cdf bounds", bool((p1 <= 1e-9).all() and (p2 >= 1 - 1e-9).all()),
               float(p1.max()), float(p2.min())))
    return ok


if __name__ == "__main__":
    for name, passed, *info in run_selftests():
        print(f"{'PASS' if passed else 'FAIL'}  {name}  {info}")
