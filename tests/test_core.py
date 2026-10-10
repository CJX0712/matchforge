# MatchForge 单元测试：核心不变量 / 确定性 / 错误码 / 配置校验。作者：晨星
from __future__ import annotations

import numpy as np
import pytest

from core.config import Config
from core.errors import ConfigValueError, MatchForgeError, error_class
from core.interfaces import is_permutation
from core.seed import set_all
from core.types import GraphPair
from data.generators import generate_correlated_pair
from eval.metrics import cycle_consistency, recovery_rate
from matching.base import alignment_score, distortion
from matching.faq import faq_solve


def _small_cfg(**kw):
    base = {
        "n_nodes": 40,
        "dgp_model": "er",
        "edge_prob": 0.5,
        "n_anchors": 5,
        "seeds": (7,),
        "noise_levels": (0.0, 0.3),
        "faq_max_iter": 30,
    }
    base.update(kw)
    return Config(**base)


def test_is_permutation_invariant():
    # is_permutation 作用于一维置换数组：元素是 0..n-1 的排列
    n = 25
    rng = np.random.default_rng(1)
    perm = rng.permutation(n)
    assert is_permutation(perm)
    # 含重复值 → 非置换
    bad = perm.copy()
    bad[1] = bad[0]
    assert not is_permutation(bad)
    # 越界值 → 非置换
    bad2 = perm.copy()
    bad2[0] = n
    assert not is_permutation(bad2)
    # 空 → 非置换
    assert not is_permutation(np.array([], dtype=int))


def test_alignment_distortion_consistency():
    # 对齐得分与失真互为对偶：‖A−PBPᵀ‖² = ‖A‖² + ‖B‖² − 2·s
    gp = generate_correlated_pair(n=30, model="er", p=0.5, noise=0.2, seed=3)
    perm = gp.true_perm
    s = alignment_score(gp.a, gp.b, perm)
    d = distortion(gp.a, gp.b, perm)
    norm_a = float((gp.a * gp.a).sum())
    norm_b = float((gp.b * gp.b).sum())
    assert abs(d - (norm_a + norm_b - 2.0 * s)) < 1e-6


def test_faq_objective_monotonic():
    # 不变量：FAQ 目标序列 s_k 单调非减（闭式线搜索保证）
    gp = generate_correlated_pair(n=30, model="er", p=0.5, noise=0.2, seed=3)
    _, _, scores = faq_solve(gp.a, gp.b, P0=None, max_iter=40, tol=1e-5, seed=3)
    for i in range(1, len(scores)):
        assert scores[i] >= scores[i - 1] - 1e-6


def test_recovery_rate_and_cycle_consistency():
    n = 20
    perm = np.arange(n)
    assert recovery_rate(perm, perm) == 1.0
    assert recovery_rate(np.flip(perm), perm) == 0.0
    assert cycle_consistency(perm) == 1.0


def test_determinism_same_seed():
    # G3：同 seed 两次运行逐位一致
    set_all(42)
    from pipeline import MatchPipeline

    cfg = _small_cfg()
    r1 = MatchPipeline(cfg).benchmark()
    r2 = MatchPipeline(cfg).benchmark()
    a = r1.aggregated["seeded"][0.3]["rec"]["mean"]
    b = r2.aggregated["seeded"][0.3]["rec"]["mean"]
    assert abs(a - b) <= 1e-12


def test_iso_recovery_gate():
    # G1：同构(η=0)下具备恢复能力的方法恢复率 ≥ 0.99
    from pipeline import MatchPipeline

    cfg = _small_cfg()
    rep = MatchPipeline(cfg).benchmark()
    for m in ("spectral", "grampa", "faq", "matchfuse", "seeded"):
        assert rep.aggregated[m][0.0]["rec"]["mean"] >= 0.99


def test_config_validation():
    with pytest.raises(ConfigValueError):
        Config(edge_prob=1.5)._validate()
    with pytest.raises(ConfigValueError):
        Config(n_nodes=0)._validate()
    with pytest.raises(ConfigValueError):
        Config(main_noise=0.99)._validate()  # 不在 noise_levels 内


def test_error_duplicate_registration():
    with pytest.raises(RuntimeError):
        error_class("E100")  # 导入期已注册


def test_error_formatting():
    e = MatchForgeError("E100", "detail")
    assert e.code == "E100"
    assert "E100" in str(e)


def test_graphpair_shape_guard():
    with pytest.raises(Exception):
        GraphPair(a=np.zeros((3, 3)), b=np.zeros((4, 4)))
