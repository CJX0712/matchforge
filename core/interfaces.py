# MatchForge · core/interfaces.py
# 匹配器协议 + 结果谓词。作者：晨星
from __future__ import annotations

from typing import Protocol

from core.types import GraphPair, MatchingResult


class Matcher(Protocol):
    name: str

    def available(self) -> bool: ...

    def match(self, gp: GraphPair) -> MatchingResult: ...


def is_permutation(perm: object) -> bool:
    arr = (
        __import__("numpy").asarray(perm, dtype=int).reshape(-1)
        if not isinstance(perm, __import__("numpy").ndarray)
        else perm.reshape(-1)
    )
    n = arr.shape[0]
    if n == 0:
        return False
    if set(arr.tolist()) != set(range(n)):
        return False
    return True
