# MatchForge · matching/spectral.py
# 谱匹配（Umeyama 1988）：邻接矩阵特征向量对齐 + 匈牙利指派。作者：晨星
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from core.types import GraphPair, MatchingResult
from matching.base import BaseMatcher, make_result, register


@register
class SpectralMatcher(BaseMatcher):
    name = "spectral"

    def available(self) -> bool:
        return True

    def _raw_perm(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n = A.shape[0]
        wA, UA = np.linalg.eigh(A)
        wB, UB = np.linalg.eigh(B)
        # 按特征值绝对值降序重排（谱匹配对符号不敏感）
        oA = np.argsort(-np.abs(wA))
        oB = np.argsort(-np.abs(wB))
        UA = UA[:, oA]
        UB = UB[:, oB]
        # 符号对齐：使每列与全局方向一致（与 ones 向量同向）
        ones = np.ones(n)
        sA = np.sign(UA.T @ ones)
        sB = np.sign(UB.T @ ones)
        UA = UA * sA[None, :]
        UB = UB * sB[None, :]
        M = UA @ UB.T  # 候选相似度
        row, col = linear_sum_assignment(-np.abs(M))
        perm = np.empty(n, dtype=np.int64)
        perm[row] = col
        return perm

    def match(self, gp: GraphPair) -> MatchingResult:
        perm = self._raw_perm(gp.a, gp.b)
        return make_result(self.name, perm, gp.a, gp.b)
