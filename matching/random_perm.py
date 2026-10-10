# MatchForge · matching/random_perm.py
# 随机置换基线（下界）。作者：晨星
from __future__ import annotations

import numpy as np

from core.types import GraphPair, MatchingResult
from matching.base import BaseMatcher, make_result, register


@register
class RandomMatcher(BaseMatcher):
    name = "random"

    def match(self, gp: GraphPair) -> MatchingResult:
        rng = np.random.default_rng(gp.seed)
        perm = rng.permutation(gp.n)
        return make_result(self.name, perm, gp.a, gp.b)
