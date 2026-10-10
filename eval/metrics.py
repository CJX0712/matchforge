# MatchForge · eval/metrics.py
# 匹配恢复率等评测指标。作者：晨星
from __future__ import annotations

import numpy as np


def recovery_rate(est_perm: np.ndarray, true_perm: np.ndarray) -> float:
    """恢复率 = 正确匹配的节点比例。"""
    est_perm = np.asarray(est_perm, dtype=np.int64).reshape(-1)
    true_perm = np.asarray(true_perm, dtype=np.int64).reshape(-1)
    return float(np.mean(est_perm == true_perm))


def cycle_consistency(perm: np.ndarray) -> float:
    """循环一致性：perm∘perm⁻¹ 应为恒等（检验置换合法性，理想为 1.0）。"""
    perm = np.asarray(perm, dtype=np.int64).reshape(-1)
    n = perm.shape[0]
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)
    return float(np.mean(perm[inv] == np.arange(n)))
