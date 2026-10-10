# MatchForge · matching/degree_profile.py
# 度画像基线：按节点度降序配对（最朴素结构线索）。作者：晨星
from __future__ import annotations

import numpy as np

from core.types import GraphPair, MatchingResult
from matching.base import BaseMatcher, make_result, register


@register
class DegreeProfileMatcher(BaseMatcher):
    name = "degree_profile"

    def match(self, gp: GraphPair) -> MatchingResult:
        deg_a = gp.a.sum(axis=1)
        deg_b = gp.b.sum(axis=1)
        # 度降序排名（稳定排序）；ties 用节点索引打破
        order_a = np.lexsort((np.arange(gp.n), -deg_a))
        order_b = np.lexsort((np.arange(gp.n), -deg_b))
        perm = np.empty(gp.n, dtype=np.int64)
        perm[order_a] = order_b
        return make_result(self.name, perm, gp.a, gp.b)
