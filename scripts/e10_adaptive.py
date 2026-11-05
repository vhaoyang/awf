"""Evaluate BQN mixture scores in quantile coordinates.

Within-component Bernstein terms use analytic moments. Cross-component
expectations use transformed quadrature, support-tail terms and numerical
inversion. Increasing M supplies a self-refinement diagnostic; agreement
between resolutions is not a rigorous error bound. No fixed wind-speed
grid is required."""
from math import comb, factorial

import numpy as np

from e10_core import BqnComponent


# ---------- Bernstein 解析矩 ----------

def J_analytic(comp: BqnComponent) -> np.ndarray:
    """单组件 J = ∫F(1−F)dx = ∫₀¹ τ(1−τ)Q'(τ)dτ 的闭式解, 返回 (n,)。"""
    K = comp.K
    m = K - 1
    d = np.diff(comp.coef, axis=1)                     # (n, K) 增量 ≥0
    jj = np.arange(m + 1)
    w = np.array([comb(m, j) * factorial(j + 1) * factorial(m - j + 1)
                  / factorial(m + 3) for j in jj])
    return K * (d * w).sum(axis=1)


def _dQ_mat(comp: BqnComponent, T: np.ndarray) -> np.ndarray:
    """Q'(τ) 于逐行 τ 矩阵 T (n, m) -> (n, m)。Q'=K Σ Δ_j B_{j,K−1}。"""
    m = comp.K - 1
    jj = np.arange(m + 1)
    wj = np.array([comb(m, j) for j in jj])
    Bm = (T[..., None] ** jj) * ((1.0 - T[..., None]) ** (m - jj)) * wj
    return comp.K * np.einsum("nmk,nk->nm", Bm, np.diff(comp.coef, axis=1))


def Q_rowwise(comp: BqnComponent, T: np.ndarray) -> np.ndarray:
    """T: (n, m) 逐行 τ 矩阵 -> Q_i(T_ij) (n, m)。"""
    B = comp._basis(T)                                  # (n, m, K+1)
    return np.einsum("nmk,nk->nm", B, comp.coef)


def F_rowwise(comp: BqnComponent, X: np.ndarray, iters: int = 60) -> np.ndarray:
    """广义逆 CDF: X (n, m) -> F_i(X_ij) (n, m), τ 域二分。"""
    lo = np.zeros(X.shape)
    hi = np.ones(X.shape)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        le = Q_rowwise(comp, mid) < X
        lo = np.where(le, mid, lo)
        hi = np.where(le, hi, mid)
    return 0.5 * (lo + hi)


# ---------- 跨组件 E|X_i−X_j| ----------

def _E_ij_panel(comp_i, comp_j, M: int) -> np.ndarray:
    """分段 Simpson 计算 E|X_i−X_j| (n,)。

    面板边界(τ 域, 每行独立): F_i(Q_j(t_k)) 与 t_k 的并集, k=0..M。
    重复边界产生零宽面板, Simpson 贡献为 0, 无需去重。
    """
    n = comp_i.coef.shape[0]
    tk = np.linspace(0.0, 1.0, M + 1)
    xj = comp_j.Q_of(tk)                                # (n, M+1)
    edges_j = F_rowwise(comp_i, xj)                     # (n, M+1)
    edges = np.sort(np.concatenate(
        [edges_j, np.tile(tk, (n, 1))], axis=1), axis=1)  # (n, 2M+2)
    # Simpson 节点: 端-中-端交错
    left, right = edges[:, :-1], edges[:, 1:]
    mid = 0.5 * (left + right)
    T = np.empty((n, 2 * edges.shape[1] - 1))
    T[:, 0::2], T[:, 1::2] = edges, mid
    Xi = Q_rowwise(comp_i, T)                           # (n, ·)
    Fj = F_rowwise(comp_j, Xi)
    g = (T * (1.0 - Fj) + Fj * (1.0 - T)) * _dQ_mat(comp_i, T)
    f0, fm, f1 = g[:, 0:-2:2], g[:, 1:-1:2], g[:, 2::2]
    h = T[:, 2::2] - T[:, 0:-2:2]
    return ((h / 6.0) * (f0 + 4.0 * fm + f1)).sum(axis=1)


def _head_tail(comp_j: BqnComponent, q0_i: np.ndarray, q1_i: np.ndarray,
               n_gl: int = 48) -> tuple:
    """x=Q_i(τ) 代换覆盖不到的两段(每行):

    head = ∫₀^{Q_i(0)} F_j(x)dx = E[(Q_i(0) − X_j)⁺]   (F_i=0 段)
    tail = ∫_{Q_i(1)}^∞ (1−F_j(x))dx = E[(X_j − Q_i(1))⁺]  (F_i=1 段)
    """
    from numpy.polynomial.legendre import leggauss
    xg, wg = leggauss(n_gl)

    def clipped_moment(q, upper):
        # ∫₀¹ (q − Q_j(t))⁺dt, 在 t*=F_j(q) 处分裂后单侧 GL
        tstar = comp_j.cdf_at(np.asarray(q, dtype=float))
        half, mid_ = 0.5 * tstar, 0.5 * tstar
        tot = np.zeros(len(q))
        for xv, wv in zip(xg, wg):
            t = mid_ + half * xv
            tot += half * wv * (q - comp_j.Q_diag(t))
        return np.maximum(tot, 0.0)

    def excess_moment(q, lower):
        # ∫₀¹ (Q_j(t) − q)⁺dt
        tstar = comp_j.cdf_at(np.asarray(q, dtype=float))
        half, mid_ = 0.5 * (1.0 - tstar), 0.5 * (1.0 + tstar)
        tot = np.zeros(len(q))
        for xv, wv in zip(xg, wg):
            t = mid_ + half * xv
            tot += half * wv * (comp_j.Q_diag(t) - q)
        return np.maximum(tot, 0.0)

    return clipped_moment(q0_i, None), excess_moment(q1_i, None)


def E_cross(comp_i: BqnComponent, comp_j: BqnComponent, M: int = 64):
    """E|X_i−X_j| (i≠j, 独立)。M 为每方向分位格数, 收敛由 M 翻倍核对。"""
    # 对称化: 两个方向的代换各算一次取平均(i 换 j), 边界信息更全
    a = _E_ij_panel(comp_i, comp_j, M)
    b = _E_ij_panel(comp_j, comp_i, M)
    ha, ta = _head_tail(comp_j, comp_i.coef[:, 0], comp_i.coef[:, -1])
    hb, tb = _head_tail(comp_i, comp_j.coef[:, 0], comp_j.coef[:, -1])
    return 0.5 * ((a + ha + ta) + (b + hb + tb))


# ---------- 三种子混合 ----------

def mixture_adaptive_scores(comps, y, M=64, check=False):
    """等权 CDF 混合的无网格评分。

    返回 dict: crps/q10/q50/q90/p_gt15/width80 (+ 可选 self_conv)。
    """
    y = np.asarray(y, dtype=float)
    Ey = np.mean([c.Ey(y) for c in comps], axis=0)
    k = len(comps)
    E = np.zeros((k, k, len(y)))
    for i in range(k):
        for j in range(k):
            if i == j:
                E[i, i] = 2.0 * J_analytic(comps[i])
            elif j > i:
                E[i, j] = E[j, i] = E_cross(comps[i], comps[j], M)
    crps = Ey - 0.5 * E.mean(axis=(0, 1))

    hi = np.maximum.reduce([c.coef[:, -1] for c in comps])
    qs = {}
    for tau in (0.1, 0.5, 0.9):
        a, b = np.zeros(len(y)), hi.copy()
        for _ in range(60):
            mid = 0.5 * (a + b)
            F = np.mean([c.cdf_at(mid) for c in comps], axis=0)
            a = np.where(F < tau, mid, a)
            b = np.where(F < tau, b, mid)
        qs[tau] = 0.5 * (a + b)
    p_gt15 = 1.0 - np.mean([c.cdf_at(np.full(len(y), 15.0)) for c in comps],
                           axis=0)
    out = dict(crps=crps, q10=qs[0.1], q50=qs[0.5], q90=qs[0.9],
               p_gt15=p_gt15, width80=qs[0.9] - qs[0.1])
    if check:
        out["self_conv"] = np.abs(
            mixture_adaptive_scores(comps, y, M=2 * M)["crps"] - crps)
    return out


def mixture_crps_only(comps, y, M=64):
    """仅混合 CRPS(供 _val_crps 与精度审计复用)。"""
    y = np.asarray(y, dtype=float)
    Ey = np.mean([c.Ey(y) for c in comps], axis=0)
    k = len(comps)
    E = 0.0
    for i in range(k):
        for j in range(k):
            E += 2.0 * J_analytic(comps[i]) if i == j else \
                E_cross(comps[i], comps[j], M)
    return Ey - 0.5 * E / (k * k)
