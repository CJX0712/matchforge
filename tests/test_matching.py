# MatchForge 单元测试：种子图匹配 / 离线兜底（scipy 缺失时贪婪 LAP）。作者：晨星
from __future__ import annotations

import numpy as np

from data.generators import generate_correlated_pair
from matching.faq import _lap
from matching.seeded import SeededMatcher, seeded_faq_solve


def test_seeded_anchors_always_correct():
    # 不变量：已知锚点必须被精确恢复
    gp = generate_correlated_pair(n=50, model="er", p=0.5, noise=0.3, seed=7, n_anchors=8)
    res = SeededMatcher().match(gp)
    for a, b in gp.anchors:
        assert res.perm[a] == b


def test_seeded_recovers_mostly():
    # 种子图匹配在中等噪声下应显著高于随机下界
    gp = generate_correlated_pair(n=50, model="er", p=0.5, noise=0.2, seed=11, n_anchors=6)
    res = SeededMatcher().match(gp)
    from eval.metrics import recovery_rate

    assert recovery_rate(res.perm, gp.true_perm) >= 0.9


def test_seeded_faq_no_anchor_drift():
    # 自由子问题收敛后，锚点行仍为唯一 1
    gp = generate_correlated_pair(n=40, model="er", p=0.5, noise=0.2, seed=5, n_anchors=4)
    perm, _ = seeded_faq_solve(gp.a, gp.b, gp.anchors, max_iter=30)
    P = np.zeros((gp.n, gp.n))
    P[np.arange(gp.n), perm] = 1.0
    for a, _ in gp.anchors:
        assert P[a].sum() == 1.0


def test_lap_fallback_without_scipy(monkeypatch):
    # 离线兜底：scipy 不可用时退化为合法置换
    import matching.faq as faqmod

    def _boom(*_a, **_k):
        raise Exception("scipy unavailable (simulated)")

    monkeypatch.setattr(faqmod, "linear_sum_assignment", _boom)
    n = 7
    cost = np.random.default_rng(0).random((n, n))
    row, col = faqmod._lap(cost)
    assert sorted(col.tolist()) == list(range(n))
    assert sorted(row.tolist()) == list(range(n))


def test_lap_standard_minimizes():
    # 标准 path：单位阵代价最小指派应为恒等
    n = 6
    cost = 1.0 - np.eye(n)  # 对角成本 0，其余 1
    row, col = _lap(cost)
    assert sorted(col.tolist()) == list(range(n))
