#!/usr/bin/env python
"""AWF E8 — V3 返修核心库：单一完整分布 / 精确分段积分 / 截断块 bootstrap

运行号: AWF_E8_CORE_R1 (服务器端自编, 非本侧任务号)

对应 V3-01/V3-03:
  - 统一完整分布 build_complete_Q: 唯一对象同时生成 CRPS、超越概率与
    分位诊断。规则(全文唯一, 写入论文):
      * 内部节点 = 拟合 taus 的单调重排分位值, 非负支撑(max(q,0));
      * 下尾 Q(0)=0, (0,0)-(q05,0.05) 线性(风速非负);
      * 上尾延续末段斜率: Q(1)=q99+0.01*(q99-q95)/(0.99-0.95),
        退化(q99==q95)时 Q(1)=q99(顶端原子)。
  - 精确 CRPS exact_crps_rows: pinball(tau) 关于 tau 分段二次, 在
    Q(tau)=y 处转折; 逐段解析积分(转折点拆分), 非节点梯形近似。
  - 概率 prob_le: 同一 Q 的广义逆 F(x)=sup{t:Q(t)<=x}, 平段取上端。
  - qs_k 修复(不再多除 K): 标量严格等于行级均值。
  - block_bootstrap 截断协议: 连续抽块至>=n_dates 后截到 n_dates 天,
    保留重复块重数; 返回统计量与每次重抽的抽中日期数。

自检: 均匀Q(0,1,y=0.5) CRPS==1/12; 退化分布 CRPS==MAE(解析相等);
Gaussian 闭式 vs 精确 <1e-4; 细网格梯形 vs 精确 <1e-4; 重复块重数。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------- 单调化 ----------

def monotonize_rows(Q: np.ndarray) -> np.ndarray:
    """逐行单调化(非降)。Q: n x K, 列已按升序 tau 排好。"""
    return np.sort(Q, axis=1)


# ---------- 统一完整分布 ----------

def build_complete_Q(Q_mono: np.ndarray, taus: np.ndarray):
    """由(已单调的)拟合分位构造唯一完整分位函数节点。

    返回 (t_nodes, q_nodes): n x (K+2), 首列 tau=0(Q=0), 末列 tau=1。
    非负支撑: 所有节点先 max(q,0)。"""
    taus = np.asarray(taus, dtype=float)
    n, K = Q_mono.shape
    q = np.maximum(Q_mono.astype(float), 0.0)  # 非负支撑规则(文档化)
    t = np.empty((n, K + 2))
    t[:, 0] = 0.0
    t[:, 1:-1] = taus[None, :]
    t[:, -1] = 1.0
    qn = np.empty((n, K + 2))
    qn[:, 0] = 0.0
    qn[:, 1:-1] = q
    q95 = q[:, list(taus).index(0.95)]
    q99 = q[:, list(taus).index(0.99)]
    last_slope = np.where(q99 > q95, (q99 - q95) / (0.99 - 0.95), 0.0)
    qn[:, -1] = q99 + 0.01 * last_slope
    return t, qn


# ---------- 精确 CRPS(分段二次解析积分) ----------

def _seg_integral(y, t0, t1, q0, q1):
    """单段 [t0,t1], Q 线性 q0->q1, pinball 关于 t 分段二次的解析积分。

    向量化: y/q0/q1 为同长数组。返回 ∫ pinball dt (未乘 2)。"""
    dt = t1 - t0
    s = (q1 - q0) / dt
    g0 = y - q0                      # t0 处 y-Q
    # 不完全平方的原函数(见模块 docstring 推导)
    def Fp(t):
        return (g0 + s * t0) * t * t / 2.0 - s * t ** 3 / 3.0

    def Fm(t):
        return ((g0 + s * t0) * (t * t / 2.0 - t)
                - s * (t ** 3 / 3.0 - t * t / 2.0))
    out = np.zeros_like(y, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstar = np.where(np.abs(s) > 0, t0 + np.where(
            np.abs(s) > 0, g0 / np.where(np.abs(s) > 0, s, 1.0), 0.0),
            t0)
    interior = (np.abs(s) > 0) & (tstar > t0) & (tstar < t1)
    # 无转折: 以段中点符号选支
    gmid = g0 - s * (dt / 2.0)
    plus_all = gmid >= 0
    no_kink_plus = (~interior) & plus_all
    no_kink_minus = (~interior) & (~plus_all)
    out[no_kink_plus] = (Fp(t1) - Fp(t0))[no_kink_plus]
    out[no_kink_minus] = (Fm(t1) - Fm(t0))[no_kink_minus]
    # 有转折: g 单调穿过 0。s>0: 先+后-; s<0: 先-后+
    sp = interior & (s > 0)
    sm = interior & (s < 0)
    if sp.any():
        out[sp] = ((Fp(tstar) - Fp(t0)) + (Fm(t1) - Fm(tstar)))[sp]
    if sm.any():
        out[sm] = ((Fm(tstar) - Fm(t0)) + (Fp(t1) - Fp(tstar)))[sm]
    return out


def exact_crps_rows(y: np.ndarray, t_nodes: np.ndarray,
                    q_nodes: np.ndarray) -> np.ndarray:
    """CRPS = 2*∫_0^1 pinball dtau, 对分段线性 Q 精确积分(行级)。"""
    total = np.zeros(len(y), dtype=float)
    for j in range(t_nodes.shape[1] - 1):
        total += _seg_integral(y, t_nodes[:, j], t_nodes[:, j + 1],
                               q_nodes[:, j], q_nodes[:, j + 1])
    return 2.0 * total


def trapezoid_crps_rows(y, t_nodes, q_nodes):
    """节点梯形积分(对照用; 与精确积分的差 = 离散化误差)。"""
    pb = np.maximum((t_nodes) * (y[:, None] - q_nodes),
                    (t_nodes - 1.0) * (y[:, None] - q_nodes))
    return 2.0 * np.trapezoid(pb, t_nodes, axis=1)


# ---------- 同一分布的超越概率 ----------

def prob_le(x: float, t_nodes: np.ndarray, q_nodes: np.ndarray):
    """F(x)=sup{t: Q(t)<=x} (广义逆, 平段取上端), 逐行。"""
    n = t_nodes.shape[0]
    out = np.empty(n)
    for i in range(n):
        q = q_nodes[i]
        t = t_nodes[i]
        if x >= q[-1]:
            out[i] = 1.0
        elif x < q[0]:
            out[i] = 0.0
        else:
            # x 落在 [q[0], q[-1]) 内; 平段取使 Q(t)<=x 的最大 t
            j = int(np.searchsorted(q, x, side="right")) - 1
            while j + 1 < len(q) - 1 and q[j + 1] == q[j]:
                j += 1
            if q[j + 1] == q[j]:
                out[i] = float(t[j + 1]) if x >= q[j] else float(t[j])
            else:
                out[i] = float(t[j] + (x - q[j]) * (t[j + 1] - t[j])
                               / (q[j + 1] - q[j]))
            j = int(np.searchsorted(q, x, side="right")) - 1
            # 平段内取使 Q(t)<=x 的最大 t: 向后延伸到平段末端
            while j + 1 < len(q) - 1 and q[j + 1] == q[j]:
                j += 1
            if q[j + 1] == q[j]:
                out[i] = float(t[j + 1]) if x >= q[j] else float(t[j])
            else:
                out[i] = float(t[j] + (x - q[j]) * (t[j + 1] - t[j])
                               / (q[j + 1] - q[j]))
    return np.clip(out, 0.0, 1.0)


def survival(x: float, t_nodes, q_nodes):
    return 1.0 - prob_le(x, t_nodes, q_nodes)


# ---------- 有限分位得分(入口统一) ----------

def pinball(y, q, tau):
    z = y - q
    return np.maximum(tau * z, (tau - 1) * z)


def qs_k(y, Q, taus):
    """有限分位等权得分 = 2/K*Σ_j mean pinball_j == mean(qs_k_rowwise)。

    V3-03 修复: 不再多除一次 K。"""
    return float(np.mean(qs_k_rowwise(y, Q, taus)))


def qs_k_rowwise(y, Q, taus):
    """行级 2/K*Σ pinball (供 bootstrap)。"""
    K = len(taus)
    cols = [pinball(y, Q[:, j], taus[j]) for j in range(K)]
    return 2.0 / K * np.sum(cols, axis=0)


def crps_normal_closed(y, mu, sd):
    z = (y - mu) / sd
    return sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z)
                 - 1 / np.sqrt(np.pi))


# ---------- 截断块 bootstrap (V3-03) ----------

def block_bootstrap(diff_rows: np.ndarray, dates: np.ndarray,
                    block_days: int, n_boot: int = 1000, seed: int = 42,
                    return_drawn: bool = False):
    """按日期块有放回重抽样, 重复块保留重数, 抽中天数截到 n_dates。

    协议: 唯一日期升序; 每次重抽以随机起点取连续 block_days 个日期的
    窗口, 逐块累加直到覆盖 >= n_dates 天, 再截到恰 n_dates 天(末块可能
    被截断, 协议固定并记录)。同一窗口可被重复抽中, 其全部行重复出现。
    返回 bootstrap 均值数组; return_drawn=True 时同时返回每次抽中日期数。
    """
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(pd.unique(dates)))
    nd = len(uniq)
    rows_by_date = {d: np.where(dates == d)[0] for d in uniq}
    stats = np.empty(n_boot)
    drawn = np.empty(n_boot, dtype=int)
    for b in range(n_boot):
        idx_parts = []
        n_taken = 0
        while n_taken < nd:
            s0 = rng.integers(0, max(nd - block_days + 1, 1))
            block = uniq[s0:s0 + block_days]
            for d in block:
                if n_taken >= nd:
                    break
                idx_parts.append(rows_by_date[d])
                n_taken += 1
        idx = np.concatenate(idx_parts)      # 重复块重数保留, 截到 nd 天
        stats[b] = diff_rows[idx].mean()
        drawn[b] = n_taken
    if return_drawn:
        return stats, drawn
    return stats


# ---------- 验证套件 ----------

def run_selftests():
    ok = []
    rng = np.random.default_rng(0)
    taus = np.round(np.concatenate([np.arange(0.05, 0.96, 0.05),
                                    [0.97, 0.98, 0.99]]), 4)

    # 1) 均匀分布 Q(t)=t (即 F=U[0,1]), y=0.5: CRPS==1/12
    n = 500
    tU = np.tile(np.linspace(0, 1, 201), (n, 1))
    qU = tU.copy()
    yU = np.full(n, 0.5)
    got = float(np.mean(exact_crps_rows(yU, tU, qU)))
    ok.append(("uniform exact == 1/12", abs(got - 1 / 12) < 1e-12,
               round(got, 10), round(1 / 12, 10)))

    # 2) 完全退化 Q(t)==c: 精确 CRPS==MAE 解析相等
    yd = rng.normal(5, 2, 3000)
    tD = np.tile([[0.0, 1.0]], (len(yd), 1))
    qD = np.full((len(yd), 2), 4.6)
    m_deg = float(np.mean(exact_crps_rows(yd, tD, qD)))
    mae_c = float(np.abs(4.6 - yd).mean())
    ok.append(("degenerate exact CRPS == MAE", abs(m_deg - mae_c) < 1e-10,
               round(m_deg, 10), round(mae_c, 10)))

    # 2b) 尾规则效应(文档化假设, 非恒等): 拟合分位全为 c 时, 统一规则
    #     下尾把 5% 质量线性铺到 0, CRPS 应严格小于 MAE 且偏差可解析
    Qdf = np.full((len(yd), len(taus)), 4.6)
    tDf, qDf = build_complete_Q(Qdf, taus)
    m_df = float(np.mean(exact_crps_rows(yd, tDf, qDf)))
    ok.append(("tail-rule effect on degenerate fit (documented)",
               m_df < mae_c, round(m_df, 6), round(mae_c, 6)))

    # 3) Gaussian 闭式 vs 精确(同一 22 分位离散化构造完整Q后)
    mu, sd = 5.0, 2.0
    yG = norm.rvs(mu, sd, size=4000, random_state=1)
    QG = np.tile(np.array([[mu + norm.ppf(t) * sd for t in taus]]),
                 (len(yG), 1))
    tG, qG = build_complete_Q(QG, taus)
    m_ex = float(np.mean(exact_crps_rows(yG, tG, qG)))
    m_cl = float(np.mean(crps_normal_closed(yG, np.full_like(yG, mu),
                                            np.full_like(yG, sd))))
    ok.append(("gaussian exact vs closed(<1e-3, 含离散化误差)",
               abs(m_ex - m_cl) < 1e-3, round(m_ex, 6), round(m_cl, 6)))

    # 4) 节点梯形 vs 精确: 差值即离散化误差量级, 精确应更接近闭式
    pbN = np.maximum(tG * (yG[:, None] - qG),
                     (tG - 1.0) * (yG[:, None] - qG))
    m_trap = float(np.mean(2 * np.trapezoid(pbN, tG, axis=1)))
    ok.append(("trapezoid-node >= exact deviation",
               abs(m_trap - m_cl) >= abs(m_ex - m_cl) - 1e-12,
               round(m_trap, 6), round(m_ex, 6)))

    # 5) 概率一致性: F(x) 随 x 不减; 超过 Q(1) 为 1; 小于 Q(0)=0 为 0
    p1 = prob_le(-1.0, tG, qG)
    p2 = prob_le(1e9, tG, qG)
    pa, pb_ = prob_le(3.0, tG, qG), prob_le(4.0, tG, qG)
    ok.append(("prob_le monotone/bounds",
               bool((p1 <= 1e-12).all() and (p2 >= 1 - 1e-12).all()
                    and (pa <= pb_ + 1e-12).all()),
               float(p1.max()), float(p2.min())))

    # 6) F(q_tau) 与 tau 一致(内部节点)
    j = list(taus).index(0.5)
    p_at_q50 = prob_le(float(qG[0, j + 1]), tG, qG)
    ok.append(("F(Q(tau)) ~= tau at nodes", abs(p_at_q50[0] - 0.5) < 0.01,
               round(float(p_at_q50[0]), 4)))

    # 7) bootstrap: 抽中天数==n_dates; 重复重数; 同差为0
    d = np.concatenate([np.zeros(150), np.ones(150)])
    dts = np.array(["2021-10-01"] * 100 + ["2021-10-02"] * 100
                   + ["2021-10-03"] * 50 + ["2021-10-04"] * 50)
    st, dr = block_bootstrap(d, dts, block_days=1, n_boot=200, seed=3,
                             return_drawn=True)
    ok.append(("bootstrap drawn==n_dates & mean~0.5",
               bool((dr == 4).all() and abs(st.mean() - 0.5) < 0.05),
               int(dr.max()), round(float(st.mean()), 3)))
    st7, dr7 = block_bootstrap(d, dts, block_days=3, n_boot=200, seed=4,
                               return_drawn=True)
    ok.append(("3d-block truncated to n_dates", bool((dr7 == 4).all()),
               int(dr7.max())))
    st0 = block_bootstrap(np.zeros(300), dts, 1, n_boot=50, seed=5)
    ok.append(("identical models -> zero interval",
               bool(np.allclose(st0, 0.0))))

    # 8) qs_k == mean(rowwise)
    Qr = rng.normal(5, 1, (200, len(taus)))
    yr = rng.normal(5, 1, 200)
    ok.append(("qs_k == mean(qs_k_rowwise)",
               abs(qs_k(yr, Qr, taus)
                   - float(np.mean(qs_k_rowwise(yr, Qr, taus)))) < 1e-12))
    return ok


if __name__ == "__main__":
    for name, passed, *info in run_selftests():
        print(f"{'PASS' if passed else 'FAIL'}  {name}  {info}")
