# MatchForge · data/__init__.py
# 数据生成包入口。作者：晨星
from __future__ import annotations

from data.generators import generate_correlated_pair, make_benchmark_suite

__all__ = ["generate_correlated_pair", "make_benchmark_suite"]
