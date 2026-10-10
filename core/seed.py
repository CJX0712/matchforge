# MatchForge · core/seed.py
# 全局确定性入口 + BLAS 线程钉死（保证同 seed 两次运行逐位一致）。作者：晨星
from __future__ import annotations

import os

# ---- BLAS / 多线程钉死：必须在 numpy 被其它模块 import 之前设好 ----
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import random as _random  # noqa: E402

import numpy as np  # noqa: E402

_GLOBAL_SEED: int | None = None


def set_all(seed: int) -> np.random.Generator:
    """一次性设齐 numpy(GlobalRNG)/legacy/Python 随机源，返回全局 Generator。"""
    global _GLOBAL_SEED
    _GLOBAL_SEED = int(seed)
    _random.seed(_GLOBAL_SEED)
    np.random.seed(_GLOBAL_SEED)
    return np.random.default_rng(_GLOBAL_SEED)


def get_seed() -> int | None:
    return _GLOBAL_SEED
