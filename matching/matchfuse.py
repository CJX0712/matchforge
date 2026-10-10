# MatchForge · matching/matchfuse.py
# MatchFuse 旗舰：异构热启动（谱/GRAMPA/度画像置换 → FAQ 多起点）+ 随机双随机重启 +
# 跨候选精确 QAP 目标（对齐得分）择优。核心思想：单算法易陷 QAP 局部极大，
# 多起点 Frank-Wolfe + 跨候选全局择优可稳定逼近真值置换。作者：晨星
from __future__ import annotations

import numpy as np

from core.types import GraphPair, MatchingResult
from matching.base import BaseMatcher, alignment_score, make_result, register
from matching.degree_profile import DegreeProfileMatcher
from matching.faq import faq_solve
from matching.grampa import GRAMPAMatcher
from matching.spectral import SpectralMatcher


def _perm_matrix(perm: np.ndarray) -> np.ndarray:
    n = perm.shape[0]
    P = np.zeros((n, n))
    P[np.arange(n), np.asarray(perm, dtype=np.int64)] = 1.0
    return P


@register
class MatchFuseMatcher(BaseMatcher):
    name = "matchfuse"

    def __init__(self, max_iter: int = 60, tol: float = 1e-5, random_inits: int = 3) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.random_inits = random_inits

    def match(self, gp: GraphPair) -> MatchingResult:
        A, B = gp.a, gp.b
        candidates: list[np.ndarray] = []

        # --- 1) 轻量启发式置换（离线兜底感知）---
        for Mat in (SpectralMatcher, GRAMPAMatcher):
            try:
                candidates.append(Mat()._raw_perm(A, B))
            except Exception:
                pass
        try:
            candidates.append(DegreeProfileMatcher().match(gp).perm)
        except Exception:
            pass

        # --- 2) FAQ 从每个启发式置换热启动（P0=置换矩阵，迭代更贴近置换流形）---
        # 注意：迭代的是 heuristics 的快照，避免遍历中 append 导致无限循环。
        heuristics = list(candidates)
        for idx, perm in enumerate(heuristics):
            try:
                P0 = _perm_matrix(perm)
                rec = faq_solve(
                    A,
                    B,
                    P0=P0,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    seed=int(gp.seed) + 17 * (idx + 1),
                )[0]
                candidates.append(rec)
            except Exception:
                pass

        # --- 3) FAQ 平凡双随机起点（等价于 FAQMatcher）---
        try:
            candidates.append(
                faq_solve(
                    A, B, P0=None, max_iter=self.max_iter, tol=self.tol, seed=int(gp.seed) + 7
                )[0]
            )
        except Exception:
            pass

        # --- 4) 额外随机双随机起点（确定性由 seed 保证）---
        rng = np.random.default_rng(int(gp.seed) + 100)
        for _ in range(self.random_inits):
            sid = int(rng.integers(1, 1 << 30))
            try:
                candidates.append(
                    faq_solve(A, B, P0=None, max_iter=self.max_iter, tol=self.tol, seed=sid)[0]
                )
            except Exception:
                pass

        # --- 5) 跨候选精确 QAP 目标（对齐得分）择优（融合核心）---
        best_perm = None
        best_score = float("-inf")
        for perm in candidates:
            s = alignment_score(A, B, perm)
            if s > best_score:
                best_score, best_perm = s, perm
        return make_result(
            self.name,
            best_perm,
            A,
            B,
            meta={"n_candidates": len(candidates), "best_score": round(best_score, 4)},
        )
