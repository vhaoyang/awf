#!/usr/bin/env python
"""Draw the BQN architecture and predictor-to-quantile transformation.

The diagram distinguishes fitting-derived preprocessing, nonnegative
Bernstein coefficients and the equal-weight mixture of three seed models.
This program renders explanatory graphics; it does not fit a model."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Polygon,
                                Circle)

plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
OUT = "outputs/manuscript_ams/figs"

fig, ax = plt.subplots(figsize=(7.2, 4.1))
ax.set_xlim(0, 100)
ax.set_ylim(0, 62)
ax.axis("off")

# ---- 马卡龙配色 ----
CREAM, CREAM_E = "#FBE8C6", "#C89A3C"     # 输入卡
GREEN, GREEN_E = "#DFF0D8", "#4F8A4C"     # 预处理卡
TEAL_F, TEAL_T, TEAL_S, TEAL_E = "#A7D3D3", "#CFE8E8", "#74AFAF", "#2F6E6E"
CORAL_F, CORAL_T, CORAL_S, CORAL_E = "#F2BCA3", "#F8D9C8", "#D68B6B", "#A8402F"
BLUE, BLUE_E = "#D8E5F4", "#3D6FB4"       # Bernstein 卡


def slab(x, y, w, h, ff, ft, fs_, ec, n_stack=3, gap=1.3):
    """2.5D 薄板(可堆叠): 前面圆角矩形+顶面/侧面平行四边形"""
    dx, dy = 2.6, 1.5
    for i in range(n_stack - 1, 0, -1):
        ax.add_patch(Polygon([(x + i * gap, y - i * gap * 0.55),
                              (x + w + i * gap, y - i * gap * 0.55),
                              (x + w + dx + i * gap, y + h + dy - i * gap * 0.55),
                              (x + dx + i * gap, y + h + dy - i * gap * 0.55)],
                             closed=True, fc=ft, ec=ec, lw=0.5, alpha=0.85))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                                fc=ff, ec=ec, lw=1.0, zorder=4))
    ax.add_patch(Polygon([(x, y + h), (x + w, y + h),
                          (x + w + dx, y + h + dy), (x + dx, y + h + dy)],
                         closed=True, fc=ft, ec=ec, lw=0.6, zorder=4))
    ax.add_patch(Polygon([(x + w, y), (x + w + dx, y + dy),
                          (x + w + dx, y + h + dy), (x + w, y + h)],
                         closed=True, fc=fs_, ec=ec, lw=0.6, zorder=4))


def card(x, y, w, h, text, fc, ec, fs=6.2, weight="normal", dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                                fc=fc, ec=ec, lw=1.0,
                                ls=(0, (4, 2)) if dashed else "-"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, linespacing=1.45, weight=weight, zorder=5)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.1, color="#333"))


# ---- 输入: 叠层卡 (F0/F1) ----
slab(3, 45, 13, 7, CREAM, "#FDF4E3", "#EBD2A0", CREAM_E, n_stack=3)
ax.text(9.5, 55.8, "F0:  13 feature columns", ha="center", fontsize=6.4,
        weight="bold")
slab(3, 25, 13, 7, CREAM, "#FDF4E3", "#EBD2A0", CREAM_E, n_stack=2)
ax.text(9.5, 36.2, "F1 = F0 + 8 terrain\n(appended)", ha="center",
        fontsize=6.0, linespacing=1.3)
# ---- 预处理卡 ----
card(21, 33, 14, 16, "preprocessing\n(train rows\nonly)\none-hot\n5-class label\n+ standardize",
     GREEN, GREEN_E, fs=5.8)
arrow(17, 48.5, 21, 44); arrow(17, 28.5, 21, 36)
# ---- 张量 x 卡 ----
card(38.5, 35, 11.5, 12, "$x\\in\\mathbb{R}^{p}$\n\n$p$ = 17 (F0)\n      25 (F1)",
     "#F4F4F4", "#777", fs=6.0)
arrow(35, 41, 38.5, 41)
# ---- BQN member 虚线分区 ----
ax.add_patch(FancyBboxPatch((52.5, 25.8), 45, 31.2,
                            boxstyle="round,pad=0.4", fc="#FBFBFD",
                            ec="#8A8A8A", lw=0.9, ls=(0, (5, 3))))
ax.text(75, 55.6, "BQN member  (single seed, deterministic GFS inputs)",
        ha="center", fontsize=6.6, weight="bold", color="#333")
# Layout: 层名入板内(图形区), 维数两行在板下空白(文字区), 无连线穿字
slab(54, 36, 8.5, 12, TEAL_F, TEAL_T, TEAL_S, TEAL_E, n_stack=3)
ax.text(58.25, 42, "FC($H_1$)", ha="center", va="center", fontsize=6.2,
        weight="bold", color="#1F4A4A", zorder=6)
ax.text(58.25, 33.8, "32 (T/T1)\n64 (S·F)", ha="center", va="top",
        fontsize=5.8, linespacing=1.25)
slab(69, 36, 8.5, 12, TEAL_F, TEAL_T, TEAL_S, TEAL_E, n_stack=2)
ax.text(73.25, 42, "FC($H_2$)", ha="center", va="center", fontsize=6.2,
        weight="bold", color="#1F4A4A", zorder=6)
ax.text(73.25, 33.8, "16 (T/T1)\n32 (S·F)", ha="center", va="top",
        fontsize=5.8, linespacing=1.25)
slab(84, 36, 8.5, 12, CORAL_F, CORAL_T, CORAL_S, CORAL_E, n_stack=2)
ax.text(88.25, 42, "$z_0,\\ldots,z_D$", ha="center", va="center",
        fontsize=6.0, weight="bold", color="#6E2417", zorder=6)
ax.text(88.25, 33.8, "$D{+}1$ = 9 / 13", ha="center", va="top",
        fontsize=5.8)
arrow(50, 41, 52.5, 41)
arrow(65.8, 43.5, 68.9, 43.5); arrow(80.4, 43.5, 83.9, 43.5)
# Preprocessing note: 分区内左下两行(避开 z 列下行线 x≈88)
ax.text(54.3, 28.0, "frozen configs:  T/T1 (32, 16), $D$ = 8",
        ha="left", va="center", fontsize=5.3, style="italic", color="#555")
ax.text(54.3, 26.7, "                        S/S1/F/F1 (64, 32), $D$ = 12",
        ha="left", va="center", fontsize=5.3, style="italic", color="#555")

# ---- Layout: z 列整列方括号 + 向量标注走廊箭头 ----
ax.plot([83.6, 83.6, 92.9, 92.9], [31.2, 30.4, 30.4, 31.2],
        color="#333", lw=1.1, solid_joinstyle="miter")  # 方括号 ⎵ 托住整列
ax.plot([88.25, 88.25, 17, 17], [30.4, 22, 22, 17.0],
        color="#333", lw=1.1, solid_capstyle="round", zorder=1)
arrow(17, 17.4, 17, 16.7)  # 末端箭头进入徽章顶部
ax.text(52, 23.4, "$z=(z_0,\\ldots,z_D)$ : full output vector",
        ha="center", va="bottom", fontsize=5.8, color="#333")

# ---- 构造链: Σsp 徽章 -> 系数卡 -> Bernstein -> Q(τ) ----
b = Circle((17, 12), 4.6, fc="#FDF1D6", ec="#B8860B", lw=1.1, zorder=5)
ax.add_patch(b)
ax.text(17, 12.9, "$\\Sigma$", ha="center", va="center", fontsize=9)
ax.text(17, 9.9, "sp", ha="center", va="center", fontsize=5.5)
ax.text(17, 4.9, "cumulative softplus\n$c_k=\\sum_{j\\leq k}$sp$(z_j)$",
        ha="center", va="top", fontsize=5.6, linespacing=1.3)
card(26, 6, 15, 12, "coefficients\n$c_0,\\ldots,c_D$\n$\\geq 0$, nondecreasing",
     "#EFE6F5", "#7A4AA0", fs=5.8)
arrow(21.8, 12, 26, 12)
card(45, 4, 24, 16, "Bernstein basis\n\n$Q(\\tau)=\\sum_{k=0}^{D} c_k"
     "\\binom{D}{k}\\tau^{k}(1-\\tau)^{D-k}$\n\nmember quantile function, "
     "$\\tau\\in[0,1]$",
     BLUE, BLUE_E, fs=6.0)
arrow(41, 12, 45, 12)
# Layout: 小图标题移到左下角内侧, 箭头从中部进入面板左缘, 互不相交
sub = ax.inset_axes([0.745, 0.045, 0.115, 0.185])
t = np.linspace(0, 1, 100)
sub.plot(t, 0.3 + 2.2 * t ** 1.6, color="#3B5285", lw=1.5)
sub.set_xticks([]); sub.set_yticks([])
sub.set_xlabel(r"$\tau$", fontsize=6, labelpad=0.5)
sub.set_ylabel(r"$Q(\tau)$", fontsize=6, labelpad=0.5)
sub.text(0.06, 0.06, "member $Q(\\tau)$\nschematic", transform=sub.transAxes,
         ha="left", va="bottom", fontsize=5.0, color="#3B5285", linespacing=1.2)
for sp in sub.spines.values():
    sp.set_linewidth(0.5)
arrow(70, 12, 73.5, 12)

plt.tight_layout()
plt.savefig(f"{OUT}/fig_bqn_arch.png", dpi=300, bbox_inches="tight")
plt.close()
print("fig_bqn_arch v3c (layout fixes) done")
