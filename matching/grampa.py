# MatchForge · matching/grampa.py
# GRAMPA (Fan, Needle & Mucha 2019)：特征间隔加权谱相似度 → 指派。作者：晨星
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from core.types import GraphPair, MatchingResult
from matching.base import BaseMatcher, make_result, register


@register
class GRAMPAMatcher(BaseMatcher):
    name = "grampa"

    def _similarity(self, A: np.ndarray, B: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        n = A.shape[0]
        wA, UA = np.linalg.eigh(A)
        wB, UB = np.linalg.eigh(B)
        ones = np.ones(n)
        fA = UA.T @ ones  # 符号歧义因子（与 UA 同号翻转 ⇒ S 不变）
        fB = UB.T @ ones
        diff = wA[:, None] - wB[None, :]
        W = 1.0 / (diff**2 + eps**2)  # 特征间隔加权
        M = W * np.outer(fA, fB)
        S = UA @ M @ UB.T
        return S

    def _raw_perm(self, A: np.ndarray, B: np.ndarray, eps: float = 1e-3) -> np.ndarray:
        n = A.shape[0]
        S = self._similarity(A, B, eps)
        row, col = linear_sum_assignment(-np.abs(S))
        perm = np.empty(n, dtype=np.int64)
        perm[row] = col
        return perm

    def match(self, gp: GraphPair) -> MatchingResult:
        perm = self._raw_perm(gp.a, gp.b)
        return make_result(self.name, perm, gp.a, gp.b, meta={"eps": 1e-3})
