#!/usr/bin/env python
"""AWF E10 — 第六轮三模型封装：BQN(torch/GPU) / QRF(sklearn) / LightGBM

运行号: AWF_E10_MODELS_R1
依据: 统一执行方案 §五(共同输入/分布构造/搜索与种子)与实测并行布局
(LGBM 6线程×4并发, QRF n_jobs=24, BQN 走 CUDA)。

- BQN: 两层 MLP(ELU) -> (K+1) Bernstein 系数, c0=softplus(a0),
  c_k=c_{k-1}+softplus(a_k) (非负非降, 整条分位函数合法); 损失=99 个
  等距内部 τ(0.01..0.99) 平均 pinball; Adam/批256/最多200轮/耐心20,
  验证指标=完整分布 CRPS(2000 行固定子样本, 终值用全验证集)。
- QRF: RandomForestRegressor + apply() 叶映射, 原生加权经验分布
  ((1/T)Σ叶内ECDF), 无任何截断。
- LightGBM: 22 分位独立模型 + e8 冻结链(排序/非负/插值/尾规则)。
- 编码差异(披露): LGBM 原生类别特征; torch/QRF 用 one-hot(terrain);
  torch 输入标准化、y 中心化定标(训练部分拟合), 系数反变换回原空间。

合法性核验 verify_component(): BQN 系数非负非降; QRF 权重和=1;
LGBM 节点非降且 Q(0)=0。分块评分 chunked_scores(): 控制内存。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/vhyang/zcode/awf/scripts")
from e10_core import (BqnComponent, LgbmComponent, QrfComponent,
                      mixture_crps, mixture_stats)
from e8_core import build_complete_Q, monotonize_rows

TAUS22 = np.round(np.concatenate([np.arange(0.05, 0.96, 0.05),
                                  [0.97, 0.98, 0.99]]), 4)
TAUS_EXT = np.concatenate([[0.0], TAUS22, [1.0]])


# ---------------- 特征编码 ----------------

def encode(df: pd.DataFrame, feats, model: str):
    """返回模型数值矩阵。LGBM 原生类别; torch/QRF one-hot terrain。"""
    if model == "lgbm":
        return df[feats]
    X = pd.get_dummies(df[feats], columns=["terrain"])
    return X.astype(np.float64).values


# ---------------- BQN ----------------

class BQN:
    """Bernstein 分位网络(torch, GPU 可用则用)。"""

    def __init__(self, degree=12, width=(64, 32), lr=1e-3, max_epochs=200,
                 patience=20, batch=256, seed=42, device=None,
                 val_sub=2000, eval_every=1):
        import torch
        self.torch = torch
        self.degree, self.width, self.lr = degree, width, lr
        self.max_epochs, self.patience, self.batch = max_epochs, patience, batch
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.val_sub, self.eval_every = val_sub, eval_every

    def fit(self, X, y, Xval=None, yval=None, verbose=False):
        t0 = time.time()
        torch = self.torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.xmu = X.mean(0)
        self.xsd = X.std(0) + 1e-8
        # 13-01 修复: 仅尺度变换 y/sigma, 不减均值 -> 标准化空间仍非负,
        # 非负 Bernstein 系数合法覆盖 [0, max(y)/sigma]; 不排除低风速
        self.ymu = 0.0
        self.ysd = float(np.std(y) + 1e-8)
        Xs = (X - self.xmu) / self.xsd
        ys = y / self.ysd
        Xt = torch.tensor(Xs, dtype=torch.float32, device=self.device)
        yt = torch.tensor(ys, dtype=torch.float32, device=self.device)
        n, d = Xs.shape
        K1 = self.degree + 1
        torch.set_num_threads(4)
        net = torch.nn.Sequential(
            torch.nn.Linear(d, self.width[0]), torch.nn.ELU(),
            torch.nn.Linear(self.width[0], self.width[1]), torch.nn.ELU(),
            torch.nn.Linear(self.width[1], K1)).to(self.device)
        self.net = net
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        taus = torch.tensor(np.linspace(0.01, 0.99, 99), dtype=torch.float32,
                            device=self.device)
        k = torch.arange(K1, device=self.device, dtype=torch.float32)
        from math import comb
        w = torch.tensor([comb(self.degree, i) for i in range(K1)],
                         dtype=torch.float32, device=self.device)
        B = (taus[:, None] ** k[None, :]) * (
            (1 - taus[:, None]) ** (self.degree - k[None, :])) * w[None, :]
        # B: (99, K+1)

        def coeffs(x):
            a = net(x)                                    # (b, K+1)
            c0 = torch.nn.functional.softplus(a[:, :1])
            inc = torch.nn.functional.softplus(a[:, 1:])
            return torch.cat([c0, c0 + torch.cumsum(inc, dim=1)], dim=1)

        val_rows = None
        if Xval is not None and yval is not None:
            idx = rng.choice(len(yval), min(self.val_sub, len(yval)),
                             replace=False)
            val_rows = (Xval[idx], yval[idx])
        best, best_state, wait = np.inf, None, 0
        self.history = []
        for ep in range(self.max_epochs):
            net.train()
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch):
                b = perm[i:i + self.batch]
                q = coeffs(Xt[b]) @ B.T                   # (b, 99)
                yy = yt[b][:, None]
                pin = torch.maximum(taus[None, :] * (yy - q),
                                    (taus[None, :] - 1) * (yy - q))
                loss = pin.mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            if val_rows is not None and ((ep + 1) % self.eval_every == 0
                                         or ep == self.max_epochs - 1):
                crps = self._val_crps(net, coeffs, val_rows)
                self.history.append(round(float(crps), 6))
                if crps < best - 1e-5:
                    best, wait = crps, 0
                    best_state = {kk: v.detach().clone()
                                  for kk, v in net.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
            if verbose and ep % 20 == 0:
                print(f"  bqn ep{ep} loss={float(loss):.5f}"
                      f" val_crps={self.history[-1] if self.history else '-'}",
                      flush=True)
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net, self.best_val_crps, self.elapsed = net, best, time.time() - t0
        return self

    def _val_crps(self, net, coeffs, val_rows):
        """验证子样本完整分布 CRPS。19-01: 与正式评分一致(块内感知网格)。"""
        Xv, yv = val_rows
        coef = self._coef(Xv)
        comp = BqnComponent(coef)
        Ey = comp.Ey(yv)
        # 19-01: BQN Q(tau) 断点(仅加密积分, 不标原子)
        from e10_core import augment_grid, SIM_GRID, _integrate_on
        bps = _bqn_breakpoints([comp])
        grid, _m = augment_grid(
            np.array([]),
            base=np.unique(np.concatenate([SIM_GRID,
                np.array(sorted(bps))] if bps else [SIM_GRID])))
        try:
            F = comp.cdf_shared(grid)
            J = _integrate_on(F, F, grid)
        except ValueError:
            # R05/V3-07: 极端支撑(>SIM_GRID 上界)时的无网格回退——单组件
            # J 用闭式 Bernstein 矩, 与正常路径同一数学量; 不影响常规折
            from e10_adaptive import J_analytic
            J = J_analytic(comp)
        return float(np.mean(Ey - 0.5 * J))

    def _coef(self, X):
        torch = self.torch
        self.net.eval()
        with torch.no_grad():
            Xs = (X - self.xmu) / self.xsd
            a = self.net(torch.tensor(Xs, dtype=torch.float32,
                                      device=self.device))
            c0 = torch.nn.functional.softplus(a[:, :1])
            inc = torch.nn.functional.softplus(a[:, 1:])
            c = torch.cat([c0, c0 + torch.cumsum(inc, 1)], 1)
        return self.ymu + self.ysd * c.cpu().numpy()

    def component(self, X) -> BqnComponent:
        return BqnComponent(self._coef(X))


# ---------------- QRF ----------------

class QRF:
    """原生分位回归森林: 分布 = (1/T)Σ_t 叶内经验分布。"""

    def __init__(self, n_estimators=500, min_samples_leaf=50,
                 max_features=1.0, seed=42, n_jobs=24):
        self.cfg = dict(n_estimators=n_estimators,
                        min_samples_leaf=min_samples_leaf,
                        max_features=max_features, n_jobs=n_jobs,
                        random_state=seed)
        self.seed = seed

    def fit(self, X, y, Xval=None, yval=None, verbose=False):
        from sklearn.ensemble import RandomForestRegressor
        t0 = time.time()
        X = np.asarray(X)
        self.rf = RandomForestRegressor(**self.cfg)
        self.rf.fit(X, y)
        self._build_leaf_tables(X, y)
        self.elapsed = time.time() - t0
        if verbose:
            print(f"  qrf fit {self.elapsed:.1f}s", flush=True)
        return self

    def _build_leaf_tables(self, X, y):
        leaf_train = self.rf.apply(X)                    # (n, T)
        T = leaf_train.shape[1]
        self.tree_leaf_samples = []
        for t in range(T):
            lut = {}
            for lid in np.unique(leaf_train[:, t]):
                lut[int(lid)] = np.sort(y[leaf_train[:, t] == lid])
            self.tree_leaf_samples.append(lut)

    def component(self, X) -> QrfComponent:
        leaf_test = self.rf.apply(np.asarray(X))
        return QrfComponent(leaf_test, self.tree_leaf_samples)


# ---------------- LightGBM (22 分位) ----------------

class LGBM22:
    """22 分位独立模型 + e8 冻结链; 6 线程; DataFrame 输入(原生类别)。"""

    def __init__(self, n_estimators=300, num_leaves=31, min_data_in_leaf=50,
                 learning_rate=0.05, seed=42, num_threads=6,
                 early_stopping=None):
        self.cfg = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                        num_leaves=num_leaves,
                        min_data_in_leaf=min_data_in_leaf,
                        subsample=0.9, subsample_freq=1,
                        colsample_bytree=0.9, seed=seed,
                        num_threads=num_threads, deterministic=True,
                        force_col_wise=True, verbosity=-1)
        self.seed = seed
        self.es = early_stopping          # None 或 (X_val, y_val) 早停
        self.best_iters = []

    def fit(self, X: pd.DataFrame, y, verbose=False):
        import lightgbm as lgb
        t0 = time.time()
        cat = [i for i, c in enumerate(X.columns) if str(X[c].dtype) == "category"]
        Q = np.empty((len(X), len(TAUS22)))
        self.best_iters, self.models = [], []
        for j, tau in enumerate(TAUS22):
            m = lgb.LGBMRegressor(objective="quantile", alpha=float(tau),
                                  **self.cfg)
            if self.es is not None:
                Xv, yv = self.es
                m.fit(X, y, categorical_feature=cat,
                      eval_set=[(Xv, yv)], eval_metric="quantile",
                      callbacks=[lgb.early_stopping(50, verbose=False)])
                Q[:, j] = m.predict(X, num_iteration=m.best_iteration_)
                self.best_iters.append(int(m.best_iteration_ or
                                           self.cfg["n_estimators"]))
            else:
                m.fit(X, y, categorical_feature=cat)
                Q[:, j] = m.predict(X)
            self.models.append(m)
        self.Q_raw, self.elapsed = Q, time.time() - t0
        return self

    def predict_Q(self, X: pd.DataFrame):
        """对新数据输出 22 分位(沿用各 booster 的最优迭代)。"""
        import lightgbm as lgb
        cat = [i for i, c in enumerate(X.columns)
               if str(X[c].dtype) == "category"]
        Q = np.empty((len(X), len(TAUS22)))
        for j, m in enumerate(self.models):
            if self.es is not None:
                Q[:, j] = m.predict(X, num_iteration=m.best_iteration_)
            else:
                Q[:, j] = m.predict(X)
        return Q

    def component(self, X=None) -> LgbmComponent:
        """X=None 用训练期已存预测; 传入 DataFrame 时对新数据预测。"""
        Q = self.Q_raw if X is None else self.predict_Q(X)
        Qm = monotonize_rows(Q)
        _, qN = build_complete_Q(Qm, TAUS22)
        return LgbmComponent(qN, TAUS_EXT)


# ---------------- 合法性核验与分块评分 ----------------

def verify_component(comp) -> dict:
    if isinstance(comp, BqnComponent):
        c = comp.coef
        return dict(type="BQN", nonneg=bool((c >= -1e-12).all()),
                    nondecreasing=bool((np.diff(c, axis=1) >= -1e-12).all()),
                    ok=bool((c >= -1e-12).all()
                            and (np.diff(c, axis=1) >= -1e-12).all()))
    if isinstance(comp, LgbmComponent):
        Q = comp.Q
        return dict(type="LGBM", q0_is_zero=bool(np.allclose(Q[:, 0], 0.0)),
                    nondecreasing=bool((np.diff(Q, axis=1) >= -1e-12).all()),
                    ok=bool(np.allclose(Q[:, 0], 0.0)
                            and (np.diff(Q, axis=1) >= -1e-12).all()))
    if isinstance(comp, QrfComponent):
        F = comp.cdf_shared(np.array([0.0, 45.0]))
        # F(0)>0 为真实零风速原子(叶内富集), 记录不判负; 合法性看 F(45)=1
        return dict(type="QRF", f0_atom=float(F[:, 0].max()),
                    f45=float(F[:, 1].min()),
                    ok=bool(abs(F[:, 1].min() - 1.0) < 1e-9))
    return dict(type="?", ok=False)


def slice_component(comp, i):
    """按行切片组件(供分块评分)。"""
    if isinstance(comp, BqnComponent):
        return BqnComponent(comp.coef[i])
    if isinstance(comp, LgbmComponent):
        return LgbmComponent(comp.Q[i], comp.taus)
    if isinstance(comp, QrfComponent):
        return QrfComponent(comp.leaf_test[i], comp.tls)
    raise TypeError(comp)


def _bqn_breakpoints(components):
    """17-01: 收集 BQN 组件 Q(tau) 关键值作为积分/反演断点。"""
    breaks = set()
    from e10_core import BqnComponent
    for c in components:
        if isinstance(c, BqnComponent):
            for tau in (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0):
                breaks.update(np.round(c.Q_of(
                    np.array([tau])).ravel(), 6).tolist())
    return breaks


def chunked_mixture_scores(components, y, chunk=2000, train_values=None):
    """混合评分: 18 号方案真分块——先切片组件与观测, 再收集块内断点。

    QRF/LGBM 走精确路径(不变); BQN/混合走块内 BQN 感知网格。
    每块独立构建积分网格(含该块的 BQN Q(tau) 关键值), 内存随块大小增长。
    train_values: QRF 原子候选。"""
    from e10_core import (QrfComponent, exact_J_qrf_qrf,
                          qrf_mixture_stats_exact)
    from e10_core import LgbmComponent, exact_J_lgbm_lgbm, \
        lgbm_mixture_stats_exact
    n = len(y)

    # 精确路径: QRF / LGBM(不变)
    if all(isinstance(c, LgbmComponent) for c in components):
        m = len(components)
        Ey = np.mean([c.Ey(y) for c in components], axis=0)
        J = np.zeros(n)
        for a in range(m):
            for b in range(m):
                J += exact_J_lgbm_lgbm(components[a], components[b])
        crps = Ey - J / (2.0 * m * m)
        st = lgbm_mixture_stats_exact(components)
        return pd.DataFrame(dict(
            crps=crps, q10=st["q10"], q50=st["q50"], q90=st["q90"],
            p_gt15=st["p_gt15"], width80=st["width80"], _grid_check=float("nan")))
    if all(isinstance(c, QrfComponent) for c in components):
        m = len(components)
        Ey = np.mean([c.Ey(y) for c in components], axis=0)
        J = np.zeros(n)
        for a in range(m):
            for b in range(m):
                J += exact_J_qrf_qrf(components[a], components[b])
        crps = Ey - J / (2.0 * m * m)
        st = qrf_mixture_stats_exact(components)
        return pd.DataFrame(dict(
            crps=crps, q10=st["q10"], q50=st["q50"], q90=st["q90"],
            p_gt15=st["p_gt15"], width80=st["width80"], _grid_check=float("nan")))

    # BQN / 混合路径: 18 号真分块 + 19-02 积分断点与原子分离
    from e10_core import (BqnComponent, augment_grid, SIM_GRID,
                          _integrate_on, generalized_quantile)
    # 长度一致性断言(防止测试用例组件行数与 y 不匹配)
    _n0 = getattr(components[0], 'coef', getattr(components[0], 'Q',
              getattr(components[0], 'leaf_test', None)))
    if _n0 is not None:
        assert _n0.shape[0] == n,             f'组件行数({_n0.shape[0]}) != y 长度({n})'
    m = len(components)
    all_crps = np.empty(n)
    all_q10 = np.empty(n)
    all_q50 = np.empty(n)
    all_q90 = np.empty(n)
    all_p15 = np.empty(n)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        comps = [slice_component(c, slice(s, e)) for c in components]
        yb = y[s:e]
        nb = e - s
        # 19-02: 积分断点(连续BQN的Q值)与真实原子(QRF/LGBM跳跃)分离
        int_breaks = set()   # 仅用于加密网格, 不标记为原子
        true_atoms = set()   # 真实的 CDF 跳跃点
        for c in comps:
            if isinstance(c, BqnComponent):
                for tau in (0.0, 0.01, 0.05, 0.25, 0.5,
                            0.75, 0.95, 0.99, 1.0):
                    int_breaks.update(np.round(c.Q_of(
                        np.array([tau])).ravel(), 6).tolist())
            if hasattr(c, "atoms"):
                kw = ({"train_values": train_values}
                      if isinstance(c, QrfComponent) else {})
                for _r, items in c.atoms(**kw).items():
                    true_atoms.update(v for v, _m in items)
        # 构建网格: 积分断点加入 base(不标原子), 真实原子走 augment_grid
        expanded_base = np.unique(np.concatenate([SIM_GRID] + (
            [np.array(sorted(int_breaks))] if int_breaks else [])))
        grid, atom_mask = augment_grid(
            np.array(sorted(true_atoms)) if true_atoms else np.array([]),
            base=expanded_base)
        # 该块的 CDF 矩阵
        Fs = [c.cdf_shared(grid) for c in comps]
        # CRPS
        Ey_b = np.mean([c.Ey(yb) for c in comps], axis=0)
        J_b = np.zeros(nb)
        for a in range(m):
            for b in range(m):
                J_b += _integrate_on(Fs[a], Fs[b], grid)
        all_crps[s:e] = Ey_b - J_b / (2.0 * m * m)
        # 统计量(同一网格, 原子掩码仅标记真实原子)
        F_mix = np.mean(Fs, axis=0)
        for tau_val, out_arr in ((0.1, all_q10), (0.5, all_q50),
                                 (0.9, all_q90)):
            out_arr[s:e] = generalized_quantile(F_mix, grid, atom_mask,
                                                tau_val)
        all_p15[s:e] = 1.0 - np.mean(
            [c.cdf_at(15.0) for c in comps], axis=0)
        del Fs, F_mix
    return pd.DataFrame(dict(
        crps=all_crps, q10=all_q10, q50=all_q50, q90=all_q90,
        p_gt15=all_p15, width80=all_q90 - all_q10, _grid_check=float("nan")))


def single_crps(comp, y, chunk=2000):
    """单组件(单种子)CRPS。19-01: BQN 用块内感知网格,与其他入口一致。"""
    from e10_core import _integrate_on, augment_grid, SIM_GRID, BqnComponent
    out = np.empty(len(y))
    for s in range(0, len(y), chunk):
        e = min(s + chunk, len(y))
        c = slice_component(comp, slice(s, e))
        Ey = c.Ey(y[s:e])
        # 19-01: BQN 组件加入自身 Q(tau) 断点(不标原子)
        if isinstance(c, BqnComponent):
            bps = _bqn_breakpoints([c])
            grid, _m = augment_grid(
                np.array([]),  # 无真实原子
                base=np.unique(np.concatenate([SIM_GRID,
                    np.array(sorted(bps))] if bps else [SIM_GRID])))
        else:
            grid = SIM_GRID
        F = c.cdf_shared(grid)
        J = _integrate_on(F, F, grid)
        out[s:e] = Ey - 0.5 * J
    return out
