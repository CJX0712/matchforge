# MatchForge · core/config.py
# 全局配置（ENV_XXX_* 覆盖 + schema 校验）。作者：晨星
from __future__ import annotations

import os
from dataclasses import dataclass

from core.errors import ConfigValueError


def _sbm_block_sizes(n: int, k: int) -> tuple[int, ...]:
    """把 n 个节点尽量均分进 k 个块（前 n%k 个块多 1 个）。"""
    base, extra = divmod(n, k)
    return tuple(base + (1 if i < extra else 0) for i in range(k))


@dataclass
class Config:
    # 合成数据
    n_nodes: int = 120
    edge_prob: float = 0.5  # 稠密 ER：节点邻域可区分 → QAP 景观唯一可恢复
    dgp_model: str = "er"  # "er"（稠密随机图，可恢复）| "sbm"（社区结构）
    sbm_blocks: int = 3
    sbm_p_in: float = 0.30  # 块内连边概率
    sbm_p_out: float = 0.02  # 块间连边概率
    n_anchors: int = 10  # 种子图匹配锚点数（旗舰方法使用）
    noise_levels: tuple = (0.0, 0.1, 0.2, 0.3, 0.4)
    seeds: tuple = (7, 11, 23, 42, 123)

    # 算法
    faq_max_iter: int = 40
    faq_tol: float = 1e-5
    fuse_random_inits: int = 1

    # 评测
    main_noise: float = 0.3  # 头条门禁所在噪声级
    gate_recovery_gain: float = 0.10  # 旗舰相对最强基线恢复率增益门槛（绝对）
    iso_recovery_min: float = 0.99  # 同构(η=0)恢复率下限

    _ENV_INTS = ("n_nodes", "faq_max_iter", "fuse_random_inits", "sbm_blocks", "n_anchors")
    _ENV_FLOATS = (
        "edge_prob",
        "faq_tol",
        "main_noise",
        "gate_recovery_gain",
        "iso_recovery_min",
        "sbm_p_in",
        "sbm_p_out",
    )

    @classmethod
    def from_env(cls, prefix: str = "MATCHFORGE_") -> Config:
        cfg = cls()
        for fname in (
            "n_nodes",
            "edge_prob",
            "dgp_model",
            "sbm_blocks",
            "sbm_p_in",
            "sbm_p_out",
            "n_anchors",
            "noise_levels",
            "seeds",
            "faq_max_iter",
            "faq_tol",
            "fuse_random_inits",
            "main_noise",
            "gate_recovery_gain",
            "iso_recovery_min",
        ):
            env_key = prefix + fname.upper()
            if env_key not in os.environ:
                continue
            raw = os.environ[env_key]
            try:
                if fname in cls._ENV_INTS:
                    setattr(cfg, fname, int(raw))
                elif fname in cls._ENV_FLOATS:
                    setattr(cfg, fname, float(raw))
                elif fname == "noise_levels":
                    setattr(cfg, fname, tuple(float(x) for x in raw.split(",") if x != ""))
                elif fname == "seeds":
                    setattr(cfg, fname, tuple(int(x) for x in raw.split(",") if x != ""))
                else:
                    setattr(cfg, fname, raw)
            except (ValueError, TypeError) as exc:
                raise ConfigValueError(f"{env_key}={raw}") from exc
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        if self.n_nodes <= 0:
            raise ConfigValueError("n_nodes must be > 0")
        if not (0.0 < self.edge_prob < 1.0):
            raise ConfigValueError("edge_prob must be in (0,1)")
        if self.faq_max_iter <= 0:
            raise ConfigValueError("faq_max_iter must be > 0")
        for nv in self.noise_levels:
            if not (0.0 <= nv < 1.0):
                raise ConfigValueError(f"noise_level {nv} must be in [0,1)")
        if self.main_noise not in self.noise_levels:
            raise ConfigValueError("main_noise must be one of noise_levels")
