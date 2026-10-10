# MatchForge · matching/base.py
# 目标函数 + 匹配器基类 + 注册表。作者：晨星
from __future__ import annotations

import numpy as np

from core.types import GraphPair, MatchingResult


def relabel(B: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """按 perm 重排 B 的行与列：B_perm[i,j] = B[perm_i, perm_j]。"""
    return B[np.ix_(perm, perm)]


def alignment_score(A: np.ndarray, B: np.ndarray, perm: np.ndarray) -> float:
    """对齐得分 s(P)=Σ A_ij B[perm_i, perm_j]（越大越好）。"""
    return float((A * relabel(B, perm)).sum())


def distortion(A: np.ndarray, B: np.ndarray, perm: np.ndarray) -> float:
    """失真 ‖A − P B Pᵀ‖²（越小越好）。"""
    return float(((A - relabel(B, perm)) ** 2).sum())


def make_result(
    name: str, perm: np.ndarray, A: np.ndarray, B: np.ndarray, meta: dict | None = None
) -> MatchingResult:
    return MatchingResult(
        name=name,
        perm=np.asarray(perm, dtype=np.int64),
        score=alignment_score(A, B, perm),
        distortion=distortion(A, B, perm),
        meta=meta or {},
    )


class BaseMatcher:
    name: str = "base"

    def available(self) -> bool:
        return True

    def match(self, gp: GraphPair) -> MatchingResult:
        raise NotImplementedError


_REGISTRY: list[type] = []


def register(cls):
    _REGISTRY.append(cls)
    return cls


def all_matchers() -> list[type]:
    return list(_REGISTRY)
