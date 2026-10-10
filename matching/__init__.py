# MatchForge · matching/__init__.py
# 匹配器包入口（base/各方法子模块）。作者：晨星
from __future__ import annotations

from matching.base import (
    BaseMatcher,
    alignment_score,
    all_matchers,
    distortion,
    make_result,
    register,
)

__all__ = [
    "BaseMatcher",
    "register",
    "all_matchers",
    "alignment_score",
    "distortion",
    "make_result",
]
