#!/usr/bin/env python
"""AWF E10 — 第六轮核心库 v2：组件分布 / 种子混合 / 混合CRPS(分解式) / 自检

运行号: AWF_E10_CORE_R1 (v2, 分解式重写; v1 网格直积版备份于 /tmp)
依据: 统一执行方案 §五/§六。正式预测 = 三种子等权 CDF 混合; 一切正式指标
由混合分布导出; CRPS 数值误差目标 <1e-4 m/s。

核心恒等式: CRPS(F_mix, y) = (1/m)Σ_c E|X_c−y| − (1/(2m²))Σ_{c,d} J_cd,
J_cd = E|X_c−X_d| = ∫ [F_c + F_d − 2 F_c F_d] dx。
- E|X−y| 逐组件精确: LGBM=分段二次解析(=2∫pinball, 复用 e8 精确积分);
  BQN=在 Q(τ*)=y 分裂点两侧 Gauss–Legendre; QRF=逐树叶 mean|s−y| 精确。
- J_cd 与 y 无关: 每折每种子对仅算一次(共享网格+逐段 Simpson), 行级
  CRPS = 逐行 Ey 减常数 —— 速度与精度同时解决, 且天然无 y 断点。

组件接口: Ey(y)->(n,) 精确 E|X−y|; cdf_shared(grid)->(n,G)。
混合工具: mixture_crps / mixture_pair_J / mixture_stats。
"""
from __future__ import annotations

import numpy as np
from scipy.special import comb as _comb

XMAX = 500.0   # 15-01: 扩大上界; 60 以上粗尾(h=2.0)以控制网格大小
# 分段网格: 质量区 [0,25] h=0.005, 中区 [25,60] h=0.05, 尾区 [60,500] h=2.0,
# 左延拓(高斯参照组件)粗(h=0.1)。真实三模型支撑非负, 左延拓仅为稳健。
NODES = np.unique(np.concatenate([
    np.arange(-5.0, 0.0, 0.1), np.arange(0.0, 25.0, 0.005),
    np.arange(25.0, 60.0, 0.05), np.arange(60.0, XMAX + 2.0, 2.0),
    [15.0]]))
MIDS = 0.5 * (NODES[:-1] + NODES[1:])
SIM_GRID = np.sort(np.concatenate([NODES, MIDS]))   # 交错: 端,中,端,...
ATOM_MASK = None   # 由 register_atoms 维护(原子节点标记, 见 mixture 评分)

# 批量评估网格 = 精细 NODES(h.01 质量区): 提速来自"单趟全矩阵 + F 复用
# (J 的 9 对与统计量共享一次求值)", 不降网格密度(粗网格实测偏差 6.7e-4
# 超标, 已弃用; 守卫断言保留)
EVAL_BASE = NODES


def _integrate_shared(Fa, Fb=None):
    """共享网格上的 ∫g dx (逐段 Simpson); g=F_a 或 F_a+F_b−2F_aF_b。"""
    g = Fa if Fb is None else Fa + Fb - 2.0 * Fa * Fb
    f0, fm, f1 = g[:, :-2:2], g[:, 1::2], g[:, 2::2]
    h = SIM_GRID[2::2] - SIM_GRID[:-2:2]
    return ((h / 6.0) * (f0 + 4.0 * fm + f1)).sum(axis=1)


def _abs_integral(y, t_nodes, q_nodes):
    """∫₀¹|Q(τ)−y|dτ 精确分段积分。t/q: (n, m) 节点(含 0/1 端)。

    段内 Q 线性: |Q−y| 为 V 形, 与 y 交点 τ* 拆分后各侧梯形精确。"""
    total = np.zeros(len(y))
    for j in range(t_nodes.shape[1] - 1):
        t0, t1 = t_nodes[:, j], t_nodes[:, j + 1]
        q0, q1 = q_nodes[:, j], q_nodes[:, j + 1]
        dt = t1 - t0
        s = (q1 - q0) / dt
        a0 = np.abs(q0 - y)
        a1 = np.abs(q1 - y)
        trap = dt * (a0 + a1) / 2.0
        # 内部穿零修正: |线性| 的梯形高估了三角形
        d0, d1 = q0 - y, q1 - y
        cross = (d0 * d1 < 0)
        if cross.any():
            tstar = t0 + (y - q0) / s[cross] * dt[cross]                 if False else t0[cross] + (0.0 - d0[cross]) / s[cross]
            # 精确面积 = 两个三角形
            h1 = tstar - t0[cross]
            h2 = t1[cross] - tstar
            exact = 0.5 * np.abs(d0[cross]) * h1 + 0.5 * np.abs(d1[cross]) * h2
            idx = np.where(cross)[0]
            trap[idx] = exact
        total += trap
    return total


# ---------------- 组件: LightGBM 22分位 ----------------

class LgbmComponent:
    """Q: (n, K+2) 完整分位函数节点(e8 build_complete_Q 语义, 含 (0,0)
    与上尾节点); taus 同形(含 0 与 1)。"""

    def __init__(self, Q, taus):
        self.Q = np.asarray(Q, dtype=np.float64)
        self.taus = np.asarray(taus, dtype=np.float64)

    def Ey(self, y):
        """E|X−y| = ∫₀¹|Q(τ)−y|dτ, 分段线性精确(Kink=Q(τ*)=y 处拆分)。"""
        tN = np.tile(self.taus, (len(y), 1))
        return _abs_integral(y, tN, self.Q)

    def cdf_shared(self, grid=None):
        grid = SIM_GRID if grid is None else grid
        self._check_support()
        F = np.empty((self.Q.shape[0], len(grid)))
        for i in range(self.Q.shape[0]):
            F[i] = np.interp(grid, self.Q[i], self.taus, left=0.0, right=1.0)
        return F

    def _check_support(self):
        m = float(np.max(self.Q))
        if m > SIM_GRID[-1]:
            raise ValueError(f"LGBM 支持上界 {m:.2f} 超过网格 {SIM_GRID[-1]}")

    def cdf_at(self, x):
        """精确 F(x) (标量 x, 逐行)。"""
        return np.array([np.interp(x, self.Q[i], self.taus, left=0.0,
                                   right=1.0)
                         for i in range(self.Q.shape[0])])

    def atoms(self, w_min=0.0, **kw):
        """平坦节点段 => 原子。返回 {row: [(value, mass), ...]}。"""
        out = {}
        for i in range(self.Q.shape[0]):
            q = self.Q[i]
            eq = np.flatnonzero(np.diff(q) <= 1e-12)
            if len(eq):
                items = [(float(q[j + 1]), float(self.taus[j + 1]
                                                 - self.taus[j]))
                         for j in eq]
                out[i] = [(v, m) for v, m in items if m > 0]
        return out


# ---------------- 组件: BQN (Bernstein 分位函数) ----------------

class BqnComponent:
    """coef: (n, K+1) 非负非降 Bernstein 系数;
    Q(τ)=Σ_k c_k C(K,k) τ^k (1−τ)^{K−k}, Q(0)=c_0, Q(1)=c_K。"""

    def __init__(self, coef):
        self.coef = np.asarray(coef, dtype=np.float64)
        self.K = self.coef.shape[1] - 1
        self.k = np.arange(self.K + 1)
        self.w = _comb(self.K, self.k)

    def _basis(self, t):
        """τ -> Bernstein 基 (len(t), K+1)。"""
        return (t[..., None] ** self.k) \
            * ((1.0 - t[..., None]) ** (self.K - self.k)) * self.w

    def Q_of(self, t):
        """t: (m,) -> Q 全矩阵 (n, m)。"""
        return self.coef @ self._basis(t).T

    def Q_diag(self, t):
        """t: (n,) -> 逐行对角 Q_i(t_i) (n,)。"""
        return np.einsum("nk,nk->n", self.coef, self._basis(t))

    def Ey(self, y, n_gl=48):
        """E|X−y| = ∫₀¹|Q(τ)−y|dτ; 在 τ*: Q(τ*)=y 处分裂后逐侧 GL。"""
        from numpy.polynomial.legendre import leggauss
        x, wgt = leggauss(n_gl)
        n = len(y)
        lo, hi = np.zeros(n), np.ones(n)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            le = self.Q_diag(mid) < y
            lo = np.where(le, mid, lo)
            hi = np.where(le, hi, mid)
        tstar = np.clip(0.5 * (lo + hi), 0.0, 1.0)
        total = np.zeros(n)
        ones = np.ones(n)
        for a, b in ((np.zeros(n), tstar), (tstar, ones)):
            half, mid_ = 0.5 * (b - a), 0.5 * (a + b)
            for xv, wv in zip(x, wgt):
                t = mid_ + half * xv
                total += half * wv * np.abs(y - self.Q_diag(t))
        return total

    def cdf_shared(self, grid=None):
        grid = SIM_GRID if grid is None else grid
        if float(self.coef[:, -1].max()) > SIM_GRID[-1]:
            raise ValueError("BQN 支持上界超过网格")
        # 16-01: tau 采样 4001 点(含 Q(0) 和 Q(1) 端点)
        t = np.linspace(0.0, 1.0, 4001)
        xq = self.Q_of(t)                            # (n, 4001) 升序(Q 单调)
        F = np.empty((self.coef.shape[0], len(grid)))
        for i in range(self.coef.shape[0]):
            F[i] = np.interp(grid, xq[i], t, left=0.0, right=1.0)
        return F

    def cdf_at(self, x):
        """精确 F(x): 每行二分求 Q(τ)=x。"""
        n = self.coef.shape[0]
        lo, hi = np.zeros(n), np.ones(n)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            le = self.Q_diag(mid) < x
            lo = np.where(le, mid, lo)
            hi = np.where(le, hi, mid)
        return 0.5 * (lo + hi)

    def atoms(self, w_min=0.0, **kw):
        return {}          # Bernstein 非负增量 => 连续分布


# ---------------- 组件: QRF (原生加权经验分布) ----------------

class QrfComponent:
    """leaf_test: (n, T) 每行每树叶子 id; tree_leaf_samples[t]:
    dict leaf_id -> 该叶训练 y 升序数组。分布 = (1/T)Σ_t LeafECDF_t。"""

    def __init__(self, leaf_test, tree_leaf_samples):
        self.leaf_test = leaf_test
        self.tls = tree_leaf_samples
        self.T = leaf_test.shape[1]

    def Ey(self, y):
        n = len(y)
        acc = np.zeros(n)
        for t in range(self.T):
            leaves = self.leaf_test[:, t]
            uniq, inv = np.unique(leaves, return_inverse=True)
            lut = self.tls[t]
            out = np.empty(n)
            for ui, lid in enumerate(uniq):
                rows = np.where(inv == ui)[0]
                s = lut[lid]
                out[rows] = np.abs(s[None, :] - y[rows][:, None]).mean(axis=1)
            acc += out
        return acc / self.T

    def cdf_shared(self, grid=None):
        grid = SIM_GRID if grid is None else grid
        n = self.leaf_test.shape[0]
        F = np.zeros((n, len(grid)))
        for t in range(self.T):
            leaves = self.leaf_test[:, t]
            uniq, inv = np.unique(leaves, return_inverse=True)
            lut = self.tls[t]
            per_leaf = np.empty((len(uniq), len(grid)))
            for ui, lid in enumerate(uniq):
                s = lut[lid]
                per_leaf[ui] = np.searchsorted(s, grid, side="right") / len(s)
            F += per_leaf[inv] / self.T
        return F

    def cdf_at(self, x):
        """精确 F(x) (标量)。"""
        return self.cdf_shared(np.array([x]))[:, 0]

    def atoms(self, w_min=0.01, train_values=None):
        """检测逐行原子(质量>=w_min): 候选=train 中重复>=50 的取值,
        行内质量 = F(v+)−F(v−) 精确计算。返回 {row: [(v, mass)]}。"""
        out = {}
        if train_values is None:
            return out
        for v in train_values:
            m = self.cdf_at(v) - self.cdf_at(np.nextafter(v, -np.inf))
            for i in np.flatnonzero(m >= w_min):
                out.setdefault(int(i), []).append((float(v),
                                                   float(m[i])))
        return out


# ---------------- QRF 精确路径(12-R1-03 关闭用) ----------------

def qrf_row_support(comp):
    """逐行原生加权支持: (vals, wts), wts=1/(T·n_leaf)。

    一次性构建并缓存于组件实例(9 个种子对复用)。"""
    if hasattr(comp, "_supcache"):
        return comp._supcache
    n, T = comp.leaf_test.shape
    out = []
    for r in range(n):
        vs, ws = [], []
        for tt in range(T):
            s = comp.tls[tt][int(comp.leaf_test[r, tt])]
            vs.append(s)
            ws.append(np.full(len(s), 1.0 / (T * len(s))))
        out.append((np.concatenate(vs), np.concatenate(ws)))
    comp._supcache = out
    return out


def exact_J_qrf_qrf(A, B, x_upper=XMAX):
    """两个 QRF 组件逐行精确 J = E|X_a−X_b|。

    合并两组加权支持排序后逐跳点分段常数积分(对阶梯精确, 无网格误差)。"""
    sa, sb = qrf_row_support(A), qrf_row_support(B)
    n = len(sa)
    J = np.empty(n)
    for r in range(n):
        va, wa = sa[r]
        vb, wb = sb[r]
        allv = np.concatenate([va, vb])
        if allv.max() > x_upper:
            raise ValueError(f"QRF 支持上界 {allv.max():.2f} 超过 {x_upper}")
        order = np.argsort(allv, kind="stable")
        wsorted = np.concatenate([wa, wb])[order]
        isA = order < len(va)
        Fa = np.cumsum(np.where(isA, wsorted, 0.0))
        Fb = np.cumsum(np.where(isA, 0.0, wsorted))
        sv = allv[order]
        d = np.diff(sv)
        g = Fa[:-1] + Fb[:-1] - 2.0 * Fa[:-1] * Fb[:-1]
        J[r] = np.dot(g, d)          # 头/尾段 F=0/1 => 贡献 0
    return J


def qrf_mixture_stats_exact(components, taus=(0.1, 0.5, 0.9), thr=15.0,
                            x_upper=XMAX):
    """QRF 种子混合的精确统计量: 池化加权 ECDF 的广义逆分位数与 P(>thr)。

    F_mix = (1/m)Σ ECDF_k = 池化样本权重/3 的 ECDF —— 分位数与阈值概率
    均为精确顺序统计量, 无网格/插值。"""
    m = len(components)
    sups = [qrf_row_support(c) for c in components]
    n = len(sups[0])
    keys = [f"q{int(round(tau * 100)):02d}" for tau in taus]
    out = {k: np.empty(n) for k in keys}
    out["p_gt15"] = np.empty(n)
    for r in range(n):
        vals = np.concatenate([s[r][0] for s in sups])
        wts = np.concatenate([s[r][1] for s in sups]) / m
        if vals.max() > x_upper:
            raise ValueError(f"支持上界 {vals.max():.2f} 超过 {x_upper}")
        order = np.argsort(vals, kind="stable")
        sv, cw = vals[order], np.cumsum(wts[order])
        for tau, k in zip(taus, keys):
            j = int(np.searchsorted(cw, tau, side="left"))
            out[k][r] = sv[min(j, len(sv) - 1)]
        k15 = int(np.searchsorted(sv, thr, side="right"))
        out["p_gt15"][r] = float(min(max(1.0 - (float(cw[k15 - 1])
                                                if k15 > 0 else 0.0), ), 0.0)
                               ) if False else min(max(
            1.0 - (float(cw[k15 - 1]) if k15 > 0 else 0.0), 0.0), 1.0)
    out["width80"] = out["q90"] - out["q10"]
    return out


def _lgbm_cdf_segments(Q_row, taus):
    """从 LGBM 分位节点提取唯一断点及 CDF 左右极限。

    返回 (uniq_vals, F_left, F_right) 其中:
    - uniq_vals: 去重后的断点(升序)
    - F_left[i]: x=uniq_vals[i] 处 CDF 左极限(开区间右端适用值)
    - F_right[i]: x=uniq_vals[i] 处 CDF 右极限(开区间左端适用值)
    相邻唯一值之间 CDF 从 F_right[i] 线性到 F_left[i+1]。
    跳跃 F_right[i]>F_left[i] 即该点原子质量。"""
    # 找到跳跃: Q 值相等的连续节点区间
    # 逐个唯一值找首次和末次出现
    uniq_vals = []
    F_left = []
    F_right = []
    i = 0
    m = len(Q_row)
    while i < m:
        j = i
        while j + 1 < m and Q_row[j + 1] == Q_row[i]:
            j += 1
        uniq_vals.append(Q_row[i])
        F_left.append(taus[i])       # 左极限 = 首次出现的 tau
        F_right.append(taus[j])      # 右极限 = 末次出现的 tau
        i = j + 1
    return np.array(uniq_vals), np.array(F_left), np.array(F_right)


def _lgbm_cdf_at(x, uniq_vals, F_left, F_right, side="right"):
    """从段表示求 CDF。side='right' 为右连续(含跳跃), 'left' 为左极限。"""
    if x < uniq_vals[0]:
        return 0.0
    if x > uniq_vals[-1]:
        return 1.0
    j = int(np.searchsorted(uniq_vals, x, side="right")) - 1
    if j < 0:
        return 0.0
    if x == uniq_vals[j]:
        return F_right[j] if side == "right" else F_left[j]
    if j + 1 >= len(uniq_vals):
        return 1.0
    # 线性插值 F_right[j] -> F_left[j+1]
    span = uniq_vals[j + 1] - uniq_vals[j]
    if span <= 0:
        return F_right[j]
    frac = (x - uniq_vals[j]) / span
    return F_right[j] + frac * (F_left[j + 1] - F_right[j])


def exact_J_lgbm_lgbm(A, B, x_lo=-5.0, x_hi=None):
    """两个 LGBM 分段线性分位组件逐行精确 J (13-02 修复版)。

    正确处理原子: 逐行取两组件唯一断点并集; 每个开区间内两个 CDF
    均线性(无跳跃), Simpson 精确; 跳跃本身零宽度不贡献积分。"""
    if x_hi is None:
        x_hi = max(float(A.Q.max()), float(B.Q.max())) + 1.0
    n = A.Q.shape[0]
    J = np.empty(n)
    for r in range(n):
        uv_a, fl_a, fr_a = _lgbm_cdf_segments(A.Q[r], A.taus)
        uv_b, fl_b, fr_b = _lgbm_cdf_segments(B.Q[r], B.taus)
        bp = np.unique(np.concatenate([uv_a, uv_b]))
        bp = bp[bp <= x_hi]
        # 每个开区间 (bp[i], bp[i+1]) 内 CDF 线性
        total = 0.0
        for i in range(len(bp) - 1):
            x0, x1 = bp[i], bp[i + 1]
            if x1 <= x0:
                continue
            # 区间内 CDF 值: 左端用 right limit, 右端用 left limit
            fa0 = _lgbm_cdf_at(x0, uv_a, fl_a, fr_a, "right")
            fa1 = _lgbm_cdf_at(x1, uv_a, fl_a, fr_a, "left")
            fb0 = _lgbm_cdf_at(x0, uv_b, fl_b, fr_b, "right")
            fb1 = _lgbm_cdf_at(x1, uv_b, fl_b, fr_b, "left")
            xm = 0.5 * (x0 + x1)
            fam = _lgbm_cdf_at(xm, uv_a, fl_a, fr_a, "right")
            fbm = _lgbm_cdf_at(xm, uv_b, fl_b, fr_b, "right")
            g0 = fa0 + fb0 - 2 * fa0 * fb0
            g1 = fa1 + fb1 - 2 * fa1 * fb1
            gm = fam + fbm - 2 * fam * fbm
            total += (x1 - x0) / 6.0 * (g0 + 4 * gm + g1)
        # 尾段: 从最后断点到 x_hi, CDF=1
        # 被积 = 1+1-2=0, 无贡献
        # 头段: x_lo 到首断点, CDF=0, 被积=0
        J[r] = total
    return J


def lgbm_mixture_stats_exact(components, taus=(0.1, 0.5, 0.9), thr=15.0,
                             x_lo=-5.0):
    """LGBM 种子混合精确统计量 (13-02 修复版)。

    分位数: 混合 CDF 的广义逆, 原子处不跨跳插值;
    P(>thr): 各组件右连续 CDF 在 thr 的值的平均。"""
    m = len(components)
    n = components[0].Q.shape[0]
    keys = [f"q{int(round(tau * 100)):02d}" for tau in taus]
    out = {k: np.empty(n) for k in keys}
    out["p_gt15"] = np.empty(n)
    for r in range(n):
        segs = [_lgbm_cdf_segments(c.Q[r], c.taus) for c in components]
        # 混合 CDF 的唯一断点
        bp = np.unique(np.concatenate([s[0] for s in segs]))
        # 在每个断点处计算混合 CDF 的左右极限
        F_l = np.zeros(len(bp))
        F_r = np.zeros(len(bp))
        for uv, fl, fr in segs:
            for i, v in enumerate(bp):
                F_l[i] += _lgbm_cdf_at(v, uv, fl, fr, "left") / m
                F_r[i] += _lgbm_cdf_at(v, uv, fl, fr, "right") / m
        # 分位数: 广义逆 F⁻¹(τ) = inf{x : F(x) ≥ τ}
        # 标准: F_r[j-1] < τ ≤ F_r[j]; 若 τ ≤ F_l[j] 则在连续段内插值,
        # 否则 τ 在原子 (F_l[j], F_r[j]] 内 -> 返回 bp[j]
        for tau, k in zip(taus, keys):
            j = int(np.searchsorted(F_r, tau, side="left"))
            if j >= len(bp):
                out[k][r] = bp[-1]
            elif j == 0:
                out[k][r] = bp[0]
            elif tau <= F_l[j]:
                f0, f1 = F_r[j - 1], F_l[j]
                if f1 <= f0:
                    out[k][r] = bp[j]
                else:
                    out[k][r] = bp[j - 1] + (tau - f0) * (bp[j] - bp[j - 1]) / (f1 - f0)
            else:
                out[k][r] = bp[j]
        # P(>thr): 各组件右连续 CDF 平均
        p = 0.0
        for uv, fl, fr in segs:
            p += _lgbm_cdf_at(thr, uv, fl, fr, "right") / m
        out["p_gt15"][r] = 1.0 - p
    out["width80"] = out["q90"] - out["q10"]
    return out


# ---------------- 参照组件(自检) ----------------

class GaussComponent:
    def __init__(self, mu, sd):
        self.mu, self.sd = mu, sd

    def Ey(self, y):
        from scipy.stats import norm
        z = (y - self.mu) / self.sd
        return self.sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z))

    def cdf_shared(self, grid=None):
        from scipy.stats import norm
        grid = SIM_GRID if grid is None else grid
        return norm.cdf((grid[None, :] - self.mu) / self.sd)   # (1, G)

    def cdf_at(self, x):
        from scipy.stats import norm
        return np.array([norm.cdf((x - self.mu) / self.sd)])

    def atoms(self, w_min=0.0, **kw):
        return {}


# ---------------- 混合工具(12-R1-02/03 修正版) ----------------

def augment_grid(atom_positions, base=None):
    """评估网格 = sort(base ∪ 原子) 再逐段取中点 -> 严格端-中-端交错。

    返回 (grid, atom_mask); mask 标记原子节点(供广义逆使用)。"""
    base = SIM_GRID if base is None else base
    atoms = np.unique(np.atleast_1d(atom_positions))
    atoms = atoms[(atoms >= base[0]) & (atoms <= base[-1])]
    U = np.unique(np.concatenate([base, atoms]))
    M = 0.5 * (U[:-1] + U[1:])
    G = np.empty(2 * len(U) - 1)
    G[0::2], G[1::2] = U, M
    mask = np.zeros(len(G), dtype=bool)
    if len(atoms):
        mask[0::2] = np.isin(U, atoms)
    return G, mask


def _integrate_on(Fa, Fb, grid):
    """逐段 Simpson 于给定交错网格(端,中,端,...)。"""
    g = Fa if Fb is None else Fa + Fb - 2.0 * Fa * Fb
    f0, fm, f1 = g[:, :-2:2], g[:, 1::2], g[:, 2::2]
    h = grid[2::2] - grid[:-2:2]
    return ((h / 6.0) * (f0 + 4.0 * fm + f1)).sum(axis=1)


def _collect_atoms(components, **kw):
    """并集各组件逐行原子 -> (全局原子位置列表, 逐行 {row:[(v,m)]})."""
    per_comp = [c.atoms(**kw) if hasattr(c, "atoms") else {} for c in components]
    vals = set()
    for d in per_comp:
        for row, items in d.items():
            for v, m in items:
                vals.add(v)
    return np.array(sorted(vals)) if vals else np.array([]), per_comp


def mixture_crps(components, y, F_cache=None, base=None, **atoms_kw):
    """分解式混合 CRPS; J_cd 逐行(随条件分布变化), 非整折常数。

    F_cache: (grid, [F_c...]) 已算缓存, 校验维度一致。
    16-01: BQN 组件的 Q(tau) 关键值纳入积分断点, 捕获窄支撑。"""
    m = len(components)
    Ey = np.mean([c.Ey(y) for c in components], axis=0)
    atoms, _ = _collect_atoms(components, **atoms_kw)
    # 16-01: BQN 支撑端点+关键分位数加入断点
    bqn_breaks = set()
    for c in components:
        if isinstance(c, BqnComponent):
            for tau in (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0):
                bqn_breaks.update(np.round(c.Q_of(
                    np.array([tau])).ravel(), 6).tolist())
    all_breaks_set = set(atoms.tolist()) if atoms.size else set()
    all_breaks_set |= bqn_breaks
    all_breaks = np.array(sorted(all_breaks_set)) if all_breaks_set else atoms
    if F_cache is not None:
        grid, Fs = F_cache
        assert len(Fs) == m and all(F.shape == (len(y), len(grid))
                                    for F in Fs), "F_cache 维度不匹配"
    else:
        grid, _mask = augment_grid(all_breaks, base=base)
        Fs = [c.cdf_shared(grid) for c in components]
    J = np.zeros_like(Ey)
    for a in range(m):
        for b in range(m):
            J += _integrate_on(Fs[a], Fs[b], grid)
    return Ey - J / (2.0 * m * m)


def generalized_quantile(F, grid, atom_mask, tau):
    """广义逆 F⁻¹(τ)=inf{x:F(x)>=τ}; 原子节点处不插值(取节点值),
    连续段线性插值。逐行向量化实现。"""
    n = F.shape[0]
    q = np.empty(n)
    for i in range(n):
        j = int(np.searchsorted(F[i], tau))
        if j <= 0:
            q[i] = grid[0]
        elif j >= len(grid):
            q[i] = grid[-1]
        elif atom_mask[j] and F[i, j] >= tau:
            q[i] = grid[j]          # 原子节点: τ 落入跳变内 -> 节点值
        else:
            f0, f1 = F[i, j - 1], F[i, j]
            q[i] = grid[j - 1] if f1 <= f0 else grid[j - 1] + (
                tau - f0) * (grid[j] - grid[j - 1]) / (f1 - f0)
    return q


def mixture_stats(components, taus=(0.1, 0.5, 0.9), thr=15.0,
                  F_cache=None, base=None, **atoms_kw):
    """混合分位数(广义逆)/超越概率(阈值处直算)/宽度。

    12-R1-02: 阈值概率由各组件 cdf_at(thr) 精确平均, 不过网格;
    分位数在含原子节点的网格上按广义逆约定反演。"""
    m = len(components)
    atoms, _ = _collect_atoms(components, **atoms_kw)
    if F_cache is not None:
        grid, Fs = F_cache
        assert all(F.shape[1] == len(grid) for F in Fs), "F_cache 维度不匹配"
    else:
        grid, _mask = augment_grid(atoms, base=base)
        Fs = [c.cdf_shared(grid) for c in components]
    _, atom_mask = augment_grid(atoms, base=base)
    F = np.mean(Fs, axis=0)
    out = {}
    for tau in taus:
        out[f"q{int(round(tau * 100)):02d}"] = generalized_quantile(
            F, grid, atom_mask, tau)
    out["p_gt15"] = 1.0 - np.mean(
        [c.cdf_at(thr) for c in components], axis=0)
    if "q10" in out and "q90" in out:
        out["width80"] = out["q90"] - out["q10"]
    return out


# ---------------- 自检 ----------------

def _Fmix_at(x, comps):
    from scipy.stats import norm
    return float(np.mean([c.cdf_shared(np.array([x]))[0, 0] for c in comps]))


def run_selftests():
    ok = []
    rng = np.random.default_rng(0)

    # 1) 高斯混合 CRPS: 分解式 vs dense 直积参照
    comps = [GaussComponent(3.0, 1.2), GaussComponent(6.0, 0.8),
             GaussComponent(8.0, 1.5)]
    y = np.array([5.3, 2.0, 9.5])
    from scipy.integrate import quad
    gd = np.linspace(-5, 20, 120001)
    Fd = np.mean([c.cdf_shared(gd) for c in comps], axis=0)
    ref = np.empty(len(y))
    for i, yi in enumerate(y):
        def _integrand(x, yi=yi):
            Fm = _Fmix_at(x, comps)
            return (Fm - (1.0 if x >= yi else 0.0)) ** 2
        ref[i] = quad(_integrand, -5, 20, limit=800,
                      points=[yi], epsabs=1e-11)[0]
    got = np.array([mixture_crps(comps, np.array([yi]))[0] for yi in y])
    ok.append(("gauss-mixture CRPS vs dense <1e-6",
               float(np.abs(got - ref).max()) < 1e-6,
               [round(float(v), 8) for v in got], [round(float(v), 8) for v in ref]))

    # 2) 退化混合 == MAE (三组件同点; sd->0 由 Gauss 极限近似)
    yv = rng.normal(5, 2, 300)
    deg = QrfComponent(np.zeros((1, 1), dtype=int),
                       {0: {0: np.array([4.605])}})
    got2 = np.array([mixture_crps([deg] * 3, np.array([yi]))[0]
                     for yi in yv])
    ref2 = np.abs(4.605 - yv)
    ok.append(("degenerate mixture ~ MAE (sd=1e-7)",
               float(np.abs(got2 - ref2).max()) < 1e-5,
               round(float(np.abs(got2 - ref2).max()), 8)))

    # 3) 混合中位数 vs dense 反演(网格 h=0.05 精度内)
    st = mixture_stats(comps, taus=(0.5,))
    ref_med = np.full(1, gd[int(np.searchsorted(Fd[0], 0.5))])
    ok.append(("mixture median vs dense (h=0.05)",
               float(np.abs(st["q50"] - ref_med).max()) < 0.06,
               round(float(np.abs(st["q50"] - ref_med).max()), 4)))

    # 4) BQN Ey vs dense 2∫pinball
    K = 12
    a = rng.normal(1, 0.5, (5, K + 1))
    c0 = np.abs(a[:, :1])
    coef = np.concatenate([c0, c0 + np.cumsum(np.abs(a[:, 1:]), axis=1)],
                          axis=1)
    bq = BqnComponent(coef)
    yb = np.array([2.0, 5.0, 8.0, 11.0, 14.0])
    td = np.linspace(0, 1, 400001)
    Qd = bq.Q_of(td)
    ref_abs = np.array([np.trapezoid(np.abs(Qd[i] - yi), td)
                        for i, yi in enumerate(yb)])
    refb = np.array([2 * np.trapezoid(
        np.maximum(td * (yi - Qd[i]), (td - 1) * (yi - Qd[i])), td)
        for i, yi in enumerate(yb)])
    gotb = bq.Ey(yb)
    ok.append(("BQN Ey(∫|Q−y|) vs dense <1e-8",
               float(np.abs(gotb - ref_abs).max()) < 1e-8,
               round(float(np.abs(gotb - ref_abs).max()), 10)))
    Fbq = bq.cdf_shared()
    from e10_core import _integrate_shared as _J2
    decb = bq.Ey(yb) - 0.5 * _J2(Fbq, Fbq)
    dmaxb = float(np.abs(decb - refb).max())
    # 阈值=方案规定的 CRPS 数值误差目标 1e-4(来源: J 的 Simpson 折点
    # 未对齐网格, 三阶误差; Ey 为精确/指数收敛)
    ok.append(("BQN decomp CRPS == dense 2∫pinball (<1e-4 spec)",
               dmaxb < 1e-4, round(dmaxb, 10)))

    # 5) LGBM Ey 自洽 + cdf_shared 单调有界
    import sys
    sys.path.insert(0, "/home/vhyang/zcode/awf/scripts")
    from e8_core import build_complete_Q, exact_crps_rows
    taus = np.round(np.concatenate([np.arange(0.05, 0.96, 0.05),
                                    [0.97, 0.98, 0.99]]), 4)
    Qr = np.sort(rng.normal(5, 1, (6, len(taus))), axis=1)
    tN, qN = build_complete_Q(Qr, taus)
    lc = LgbmComponent(qN, tN[0])
    yl = rng.normal(5, 2, 6)
    Fc = lc.cdf_shared()
    mono = bool((np.diff(Fc, axis=1) >= -1e-12).all()
                and (Fc >= -1e-12).all() and (Fc <= 1 + 1e-12).all())
    # 交叉验证: 分解式单分布 CRPS(Ey−½J_aa) 必须等于 e8 的 2∫pinball
    from e10_core import _integrate_shared as _J
    decomp = lc.Ey(yl) - 0.5 * _J(Fc, Fc)
    pinb = exact_crps_rows(yl, tN, qN)
    dmax = float(np.abs(decomp - pinb).max())
    ok.append(("LGBM decomp CRPS == e8 2∫pinball (<1e-4 spec) & CDF monotone",
               dmax < 1e-4 and mono, round(dmax, 10)))
    # LGBM Ey vs dense ∫|Q−y|
    td = np.linspace(0, 1, 400001)
    qd = np.array([np.interp(td, tN[i], qN[i]) for i in range(len(yl))])
    refEy = np.array([np.trapezoid(np.abs(qd[i] - yi), td)
                      for i, yi in enumerate(yl)])
    dEy = float(np.abs(lc.Ey(yl) - refEy).max())
    ok.append(("LGBM Ey == dense ∫|Q−y|", dEy < 1e-6, round(dEy, 10)))

    # 6) QRF Ey/cdf 小例暴力对照
    s1 = {0: {10: np.array([1.0, 2.0, 3.0]), 20: np.array([5.0])}}
    lt = np.array([[10], [20]])
    qc = QrfComponent(lt, s1)
    gotq = qc.Ey(np.array([2.5, 2.5]))
    Fq = qc.cdf_shared(np.array([1.5, 2.5, 4.0, 5.0]))
    refF = np.array([[1 / 3, 2 / 3, 1.0, 1.0],
                     [0.0, 0.0, 0.0, 1.0]])
    ok.append(("QRF Ey & CDF brute force",
               float(np.abs(gotq - np.array([5.0 / 6.0, 2.5])).max()) < 1e-12
               and float(np.abs(Fq - refF).max()) < 1e-12,
               [round(float(v), 4) for v in gotq]))

    # ---- 12-R1 评审解析反例 ----
    taus24 = TAUS_EXT if "TAUS_EXT" in dir() else np.concatenate(
        [[0.0], np.round(np.concatenate([np.arange(0.05, 0.96, 0.05),
                                         [0.97, 0.98, 0.99]]), 4), [1.0]])

    # A. Uniform(0,2), y=1: Ey=0.5, CRPS=1/6 —— LGBM/BQN 两种表示,
    #    单组件与三同组件混合一致
    q_unif = 2.0 * taus24
    lc_u = LgbmComponent(np.tile(q_unif, (1, 1)), taus24)
    K = 12
    coef_u = np.array([[2.0 * k / K for k in range(K + 1)]])
    bq_u = BqnComponent(coef_u)
    for name, c in (("LGBM", lc_u), ("BQN", bq_u)):
        yu = np.array([1.0])
        ey = float(c.Ey(yu)[0])
        crps1 = float(mixture_crps([c], yu)[0])
        crps3 = float(mixture_crps([c, c, c], yu)[0])
        ok.append((f"12R1-A Uniform(0,2) {name}: Ey=0.5 CRPS=1/6 单==混合",
                   abs(ey - 0.5) < 1e-8 and abs(crps1 - 1 / 6) < 1e-6
                   and abs(crps3 - crps1) < 1e-9,
                   round(ey, 8), round(crps1, 8), round(crps3, 8)))

    # B. 点质量 4.605, y=2: Ey=CRPS=2.605 —— 三种表示
    lc_p = LgbmComponent(np.full((1, len(taus24)), 4.605), taus24)
    bq_p = BqnComponent(np.full((1, K + 1), 4.605))
    qc_p = QrfComponent(np.zeros((1, 1), dtype=int),
                        {0: {0: np.array([4.605])}})
    for name, c in (("LGBM", lc_p), ("BQN", bq_p), ("QRF", qc_p)):
        yp = np.array([2.0])
        ey = float(c.Ey(yp)[0])
        crps3 = float(mixture_crps([c] * 3, yp)[0])
        ok.append((f"12R1-B 点质量4.605 {name}: Ey=CRPS=2.605",
                   abs(ey - 2.605) < 1e-9 and abs(crps3 - 2.605) < 1e-7,
                   round(ey, 8), round(crps3, 8)))

    # C. 两点 {0.013,1.017} 各半(QRF 两树): CRPS(y=.5)=0.251 精确;
    #    平移 {20.013,21.017} y=20.5 同值; 广义逆 q50=0.013
    def two_point(v1, v2):
        return QrfComponent(
            np.array([[0, 1]]),
            {0: {0: np.array([v1]), 1: np.array([v2])},
             1: {0: np.array([v1]), 1: np.array([v2])}})
    c2 = two_point(0.013, 1.017)
    r1 = float(mixture_crps([c2, c2, c2], np.array([0.5]),
                            train_values=[0.013, 1.017])[0])
    c2b = two_point(20.013, 21.017)
    r2 = float(mixture_crps([c2b, c2b, c2b], np.array([20.5]),
                            train_values=[20.013, 21.017])[0])
    st2 = mixture_stats([c2] * 3, train_values=[0.013, 1.017])
    ok.append(("12R1-C 两点分布 CRPS=0.251(两处) & q50=0.013",
               abs(r1 - 0.251) < 1e-4 and abs(r2 - 0.251) < 1e-4
               and abs(float(st2["q50"][0]) - 0.013) < 1e-9,
               round(r1, 6), round(r2, 6), float(st2["q50"][0])))

    # D. 零风速点质量: q10=q50=q90=0, 宽度 0
    c0 = QrfComponent(np.zeros((1, 1), dtype=int),
                      {0: {0: np.array([0.0])}})
    st0 = mixture_stats([c0] * 3, train_values=[0.0])
    ok.append(("12R1-D 点质量在0: 分位数全0(非负中位数)",
               float(st0["q10"][0]) == 0.0 and float(st0["q50"][0]) == 0.0
               and float(st0["q90"][0]) == 0.0 and float(st0["width80"][0]) == 0.0,
               [float(st0[k][0]) for k in ("q10", "q50", "q90")]))

    # E. 支持上界压力: {44,46} y=45 -> CRPS=0.5; 点质量50 分位数=50
    c4446 = two_point(44.0, 46.0)
    r4446 = float(mixture_crps([c4446] * 3, np.array([45.0]),
                               train_values=[44.0, 46.0])[0])
    c50 = QrfComponent(np.zeros((1, 1), dtype=int),
                       {0: {0: np.array([50.0])}})
    st50 = mixture_stats([c50] * 3, train_values=[50.0])
    ok.append(("12R1-E 边界: {44,46}y=45 CRPS=0.5; 点质量50 q50=50",
               abs(r4446 - 0.5) < 1e-4 and abs(float(st50["q50"][0]) - 50.0) < 1e-9,
               round(r4446, 6), float(st50["q50"][0])))
    return ok


if __name__ == "__main__":
    for name, passed, *info in run_selftests():
        print(f"{'PASS' if passed else 'FAIL'}  {name}  {info}")
