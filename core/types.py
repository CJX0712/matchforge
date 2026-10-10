# MatchForge · core/types.py
# 领域数据结构。作者：晨星
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GraphPair:
    """一对待匹配图：A、B 为对称 0/1 邻接矩阵；true_perm[i] = A 的节点 i 对应 B 的节点 true_perm[i]。"""

    a: np.ndarray
    b: np.ndarray
    true_perm: np.ndarray | None = None
    seed: int = 0
    name: str = ""
    noise: float = 0.0
    anchors: list[tuple] | None = None  # 已知对应 [(a_idx, b_idx), ...]（种子图匹配用）

    def __post_init__(self) -> None:
        self.a = np.asarray(self.a, dtype=np.float64)
        self.b = np.asarray(self.b, dtype=np.float64)
        if self.a.shape != self.b.shape:
            from core.errors import GraphDataError

            raise GraphDataError("A/B shape mismatch", self.a.shape, self.b.shape)
        if self.a.ndim != 2 or self.a.shape[0] != self.a.shape[1]:
            from core.errors import GraphDataError

            raise GraphDataError("A/B must be square (n,n)")
        self.n: int = self.a.shape[0]

    @property
    def is_isomorphic(self) -> bool:
        return bool(self.noise == 0.0)


@dataclass
class MatchingResult:
    """单个匹配器输出：perm 为长度 n 的置换数组（perm[i]=B 中匹配 A 节点 i 的节点）。"""

    name: str
    perm: np.ndarray
    score: float = 0.0  # 对齐得分 s(P)=Σ A_ij B[perm_i,perm_j]（越大越好）
    distortion: float = 0.0  # 失真 ‖A − P B Pᵀ‖²（越小越好）
    n: int = 0
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.perm = np.asarray(self.perm, dtype=np.int64).reshape(-1)
        self.n = int(self.perm.shape[0])
