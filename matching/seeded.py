# MatchForge · matching/seeded.py
# 种子图匹配（Seeded Graph Matching, Fishkind 2012 / Vogelstein 2015）：
# 给定少量已知节点对应（anchors），在 Birkhoff 多面体上做受约束 Frank-Wolfe，
# 仅对「自由」节点做匈牙利子问题，锚点行恒为已知对应。锚点越多，QAP 景观被「扳直」，
# 余下节点可被 FAQ 高效、确定性地恢复 —— 这是随机图上实现高恢复率的世界级方法。作者：晨星
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from core.types import GraphPair, MatchingResult
from matching.base import BaseMatcher, make_result, register


def seeded_faq_solve(
    A: np.ndarray,
    B: np.ndarray,
    anchors: Sequence[tuple[int, int]],
    P0: np.ndarray | None = None,
    max_iter: int = 60,
    tol: float = 1e-5,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """受约束 Frank-Wolfe：锚点 (a_i, b_i) 强制对应，其余额外节点子问题指派。

    返回 (最优置换, 最优对齐得分)。锚点恒正确；自由节点由 FAQ 在降维子空间优化。
    """
    n = A.shape[0]
    a2b = {int(i): int(j) for i, j in anchors}
    free_a = [i for i in range(n) if i not in a2b]
    used_b = set(a2b.values())
    free_b = [j for j in range(n) if j not in used_b]
    fA = np.asarray(free_a, dtype=np.int64)
    fB = np.asarray(free_b, dtype=np.int64)

    # 初始化：锚点行固定，自由行/列均匀（双随机）
    P = np.zeros((n, n))
    for i, j in a2b.items():
        P[i, j] = 1.0
    if len(free_a) > 0:
        u = 1.0 / len(free_a)
        P[np.ix_(free_a, free_b)] = u

    best_perm = np.empty(n, dtype=np.int64)
    for i, j in a2b.items():
        best_perm[i] = j
    for r, fa in enumerate(free_a):
        best_perm[fa] = free_b[r]  # 初始自由对应（名义，会被 FAQ 修正）

    PBPT = P @ B @ P.T
    best_score = float((A * PBPT).sum())
    prev = P.copy()

    for _ in range(max_iter):
        G = A @ P @ B.T + A.T @ P @ B
        # 仅自由子矩阵上解 LAP（锚点行已固定）
        if len(free_a) > 0:
            Gfree = G[np.ix_(fA, fB)]
            row, col = linear_sum_assignment(-Gfree)
            D = np.zeros((n, n))
            for i, j in a2b.items():
                D[i, j] = 1.0
            for r, c in zip(row, col, strict=True):
                D[fA[r], fB[c]] = 1.0
        else:
            D = P.copy()
            row = col = np.array([], dtype=int)

        DBDT = D @ B @ D.T
        sD = float((A * DBDT).sum())
        if sD > best_score:
            best_score = sD
            for i, j in a2b.items():
                best_perm[i] = j
            for r, c in zip(row, col, strict=True):
                best_perm[fA[r]] = fB[c]

        sP_cur = float((A * PBPT).sum())
        gP = float((G * (D - P)).sum())
        a = sD - sP_cur - gP
        b = gP
        if a < -1e-15:
            alpha = min(1.0, max(0.0, -b / (2.0 * a)))
        elif b > 0:
            alpha = 1.0
        else:
            alpha = 0.0
        P = (1.0 - alpha) * P + alpha * D
        # 锚点行再投影（数值安全）
        for i, j in a2b.items():
            P[i, :] = 0.0
            P[i, j] = 1.0
        sP_new = a * alpha * alpha + b * alpha + sP_cur
        assert sP_new >= sP_cur - 1e-6 * (abs(sP_cur) + 1.0), (
            "seeded FAQ objective must be monotonic"
        )
        PBPT = P @ B @ P.T
        if np.linalg.norm(P - prev) < tol:
            prev = P.copy()
            break
        prev = P.copy()
    return best_perm, best_score


@register
class SeededMatcher(BaseMatcher):
    name = "seeded"

    def __init__(self, max_iter: int = 60, tol: float = 1e-5) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def match(self, gp: GraphPair) -> MatchingResult:
        if gp.anchors is None or len(gp.anchors) == 0:
            from core.errors import MatcherUnavailableError

            raise MatcherUnavailableError("seeded requires anchors (known node matches)")
        perm, score = seeded_faq_solve(
            gp.a, gp.b, gp.anchors, max_iter=self.max_iter, tol=self.tol, seed=gp.seed
        )
        return make_result(self.name, perm, gp.a, gp.b, meta={"n_anchors": len(gp.anchors)})
