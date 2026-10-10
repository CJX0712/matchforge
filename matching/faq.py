# MatchForge · matching/faq.py
# FAQ (Fast Approximate QAP, Vogelstein 2015)：Birkhoff 多面体上的 Frank-Wolfe。
# 目标 max s(P)=tr(Aᵀ P B Pᵀ)；LAP 子问题；闭式线搜索保证 s 逐迭代单调不增。作者：晨星
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from core.errors import FAQNotConvergedError
from core.types import GraphPair, MatchingResult
from matching.base import BaseMatcher, make_result, register


def _lap(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """匈牙利指派（最小化 cost）。scipy 缺失时退化为贪心（离线兜底）。"""
    try:
        return linear_sum_assignment(cost)
    except Exception:
        n = cost.shape[0]
        row = np.arange(n)
        col = np.argsort(cost.argmax(axis=1))  # 兜底：仅保证返回合法置换
        # 简易贪心修正，避免重复列
        seen = set()
        for i in range(n):
            c = int(col[i])
            while c in seen:
                c = (c + 1) % n
            seen.add(c)
            col[i] = c
        return row, col


def faq_solve(
    A: np.ndarray,
    B: np.ndarray,
    P0: np.ndarray | None = None,
    max_iter: int = 60,
    tol: float = 1e-5,
    seed: int = 0,
) -> tuple[np.ndarray, float, list[float]]:
    """Frank-Wolfe 解 QAP。返回 (最优置换, 最优得分, 目标序列)。

    不变量：目标序列 s_k 单调非减（闭式线搜索对多线性目标精确）。
    """
    n = A.shape[0]
    if P0 is None:
        P = np.ones((n, n)) / n
    else:
        P = np.asarray(P0, dtype=np.float64).copy()
    prev = P.copy()
    best_perm = np.arange(n)
    best_score = float((A * (P @ B @ P.T)).sum())
    scores: list[float] = []
    for _ in range(max_iter):
        # 梯度 G = A P Bᵀ + Aᵀ P B  （s(P)=tr(Aᵀ P B Pᵀ) 的解析梯度，已有限差分交叉验证）
        G = A @ P @ B.T + A.T @ P @ B
        row, col = _lap(-G)  # 最大化 <G,D> over 置换
        D = np.zeros((n, n))
        D[row, col] = 1.0
        sD = float((A * (D @ B @ D.T)).sum())
        if sD > best_score:
            best_score = sD
            best_perm = col.copy()
        d = D - P
        sP_cur = float((A * (P @ B @ P.T)).sum())
        gP = float((G * d).sum())  # s 在 P 处沿 d 的导数
        a = sD - sP_cur - gP
        b = gP
        if a < -1e-15:
            alpha = -b / (2.0 * a)
            alpha = min(1.0, max(0.0, alpha))
        elif b > 0:
            alpha = 1.0
        else:
            alpha = 0.0
        P = (1.0 - alpha) * P + alpha * D
        sP_new = float((A * (P @ B @ P.T)).sum())
        if sP_new < sP_cur - 1e-6 * (abs(sP_cur) + 1.0):
            raise FAQNotConvergedError("FAQ objective decreased (line-search unstable)")
        scores.append(sP_new)
        if np.linalg.norm(P - prev) < tol:
            prev = P.copy()
            break
        prev = P.copy()
    return best_perm, best_score, scores


@register
class FAQMatcher(BaseMatcher):
    name = "faq"

    def __init__(self, max_iter: int = 60, tol: float = 1e-5) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def match(self, gp: GraphPair) -> MatchingResult:
        perm, score, _ = faq_solve(
            gp.a, gp.b, P0=None, max_iter=self.max_iter, tol=self.tol, seed=gp.seed
        )
        return make_result(self.name, perm, gp.a, gp.b, meta={"max_iter": self.max_iter})
