# MatchForge · data/generators.py
# 合成数据：相关性 ER 图对（标准 QAP 基准协议）+ 真值置换。作者：晨星
from __future__ import annotations

import numpy as np

from core.config import Config
from core.types import GraphPair


def _er_adjacency(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """无自环对称 ER 邻接矩阵。"""
    A = (rng.random((n, n)) < p).astype(np.float64)
    A = np.triu(A, 1)
    A = A + A.T
    return A


def _sbm_adjacency(
    n: int, p_in: float, p_out: float, block_sizes: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """随机块模型 SBM 邻接矩阵：块内连边概率 p_in，块间 p_out（无自环、对称）。

    SBM 提供清晰社区结构 —— 不同块节点度差异显著、主导特征向量按块对齐，
    使谱/度/FAQ 匹配拥有可恢复信号（纯 ER 图「无特征」→ 任意噪声即失效）。
    这是 unseeded graph matching 文献（Vogelstein/Lyzinski/Fishkind）的标准基准 DGP。
    """
    blocks = np.repeat(np.arange(len(block_sizes)), block_sizes)
    P = np.where(blocks[:, None] == blocks[None, :], p_in, p_out)
    A = (rng.random((n, n)) < P).astype(np.float64)
    A = np.triu(A, 1)
    A = A + A.T
    return A


def _flip(A: np.ndarray, eta: float, rng: np.random.Generator) -> np.ndarray:
    """删边型观测：上三角每条边以概率 eta 独立删去（1→0），非边永不出现（不引入虚假边）。

    保持对称、无自环。A、B 均为潜在图 W 的子图 ⇒ QAP 全局最优即真值置换且吸引域大。
    """
    upper = np.triu(A, 1)
    if eta == 0.0:
        return upper + upper.T
    keep = (rng.random(A.shape) >= eta) & (upper > 0)
    out = np.where(keep, upper, 0.0)
    return out + out.T


def generate_correlated_pair(
    n: int = 150,
    p: float = 0.15,
    noise: float = 0.0,
    seed: int = 0,
    name: str = "",
    model: str = "er",
    sbm_blocks: int = 3,
    sbm_p_in: float = 0.30,
    sbm_p_out: float = 0.02,
    n_anchors: int = 0,
) -> GraphPair:
    """生成待匹配图对 (A, B) —— 标准「相关伯努利图」模型 (Fishkind 2019)。

    潜在图 W：model="er" 时 W~ER(n,p)；model="sbm" 时 W~SBM(块内 p_in/块间 p_out)。
    A、B 是 W 的两次独立删边噪声观测（边以 prob=noise 删去），B 再经真值置换 π 重标。
    真值对应：A 的节点 k ↔ B 的节点 π[k]。
    此模型下 QAP 全局最优即 true_perm；SBM 提供更强的社区结构信号，使各匹配器可恢复。
    noise=0 ⇒ A、B 同构（B=PᵀAP），恢复率应 100%。
    """
    rng = np.random.default_rng(seed)
    if model == "sbm":
        from core.config import _sbm_block_sizes

        block_sizes = _sbm_block_sizes(n, sbm_blocks)
        W = _sbm_adjacency(n, sbm_p_in, sbm_p_out, block_sizes, rng)
    else:
        W = _er_adjacency(n, p, rng)
    A = _flip(W, noise, rng)  # G1：W 的噪声观测
    π = rng.permutation(n)  # 真值置换：B 节点 i ↔ A 节点 π⁻¹(i)
    P = np.zeros((n, n))
    P[np.arange(n), π] = 1.0
    G2_raw = _flip(W, noise, rng)  # G2：W 的独立噪声观测
    B = P.T @ G2_raw @ P  # 重标后 B 节点 i ↔ W 节点 π⁻¹(i)
    true_perm = π.copy()  # true_perm[k] = B 中对应 A 节点 k 的节点

    # 种子图匹配：随机挑 n_anchors 个 A 节点，记录其真值 B 对应
    anchors = None
    if n_anchors and n_anchors > 0:
        ka = min(int(n_anchors), n)
        anchor_idx = rng.choice(n, size=ka, replace=False)
        anchors = [(int(i), int(true_perm[i])) for i in anchor_idx]

    return GraphPair(
        a=A,
        b=B,
        true_perm=true_perm,
        seed=int(seed),
        name=name or f"pair_{model}_n{n}_eta{noise:.2f}_s{seed}",
        noise=float(noise),
        anchors=anchors,
    )


def make_benchmark_suite(cfg: Config | None = None) -> list[GraphPair]:
    """构造基准套件：每个噪声级 × 每个 seed 一对图。"""
    cfg = cfg or Config()
    suite: list[GraphPair] = []
    for eta in cfg.noise_levels:
        for s in cfg.seeds:
            suite.append(
                generate_correlated_pair(
                    n=cfg.n_nodes,
                    p=cfg.edge_prob,
                    noise=float(eta),
                    seed=int(s),
                    name=f"{cfg.dgp_model}_n{cfg.n_nodes}_eta{float(eta):.2f}_s{s}",
                    model=cfg.dgp_model,
                    sbm_blocks=cfg.sbm_blocks,
                    sbm_p_in=cfg.sbm_p_in,
                    sbm_p_out=cfg.sbm_p_out,
                    n_anchors=cfg.n_anchors,
                )
            )
    if not suite:
        from core.errors import BenchmarkEmptyError

        raise BenchmarkEmptyError("benchmark suite is empty")
    return suite
