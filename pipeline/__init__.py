# MatchForge · pipeline/pipeline.py
# 端到端链路：对图对运行所有匹配器 → 评测 → 跨 seed/噪声聚合。作者：晨星
from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field

import numpy as np

from core.config import Config
from core.types import GraphPair, MatchingResult
from data.generators import make_benchmark_suite
from eval.metrics import recovery_rate

_METHODS_LOCAL = {
    "random": "matching.random_perm.RandomMatcher",
    "degree_profile": "matching.degree_profile.DegreeProfileMatcher",
    "spectral": "matching.spectral.SpectralMatcher",
    "faq": "matching.faq.FAQMatcher",
    "grampa": "matching.grampa.GRAMPAMatcher",
    "matchfuse": "matching.matchfuse.MatchFuseMatcher",
    "seeded": "matching.seeded.SeededMatcher",
}


def _load(spec: str):
    mod, cls = spec.rsplit(".", 1)
    return getattr(importlib.import_module(mod), cls)


def build_methods(cfg: Config) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, spec in _METHODS_LOCAL.items():
        try:
            cls = _load(spec)
            if key == "faq":
                out[key] = cls(max_iter=cfg.faq_max_iter, tol=cfg.faq_tol)
            elif key == "matchfuse":
                out[key] = cls(
                    max_iter=cfg.faq_max_iter, tol=cfg.faq_tol, random_inits=cfg.fuse_random_inits
                )
            else:
                out[key] = cls()
        except Exception:
            continue
    return out


@dataclass
class BenchmarkRow:
    method: str
    noise: float
    seed: int
    n: int
    recovery: float
    score: float
    distortion: float
    elapsed_sec: float = 0.0


@dataclass
class PipelineReport:
    rows: list[BenchmarkRow] = field(default_factory=list)
    aggregated: dict[str, dict[float, dict[str, float]]] = field(default_factory=dict)


class MatchPipeline:
    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self.methods = build_methods(self.cfg)

    def run_pair(self, gp: GraphPair) -> dict[str, MatchingResult]:
        results: dict[str, MatchingResult] = {}
        for name, matcher in self.methods.items():
            t0 = time.perf_counter()
            try:
                res = matcher.match(gp)
            except Exception as exc:  # 单个方法失败不应拖垮整条链路
                from core.errors import MatcherUnavailableError

                raise MatcherUnavailableError(name) from exc
            dt = time.perf_counter() - t0
            res.meta.setdefault("elapsed_sec", round(dt, 6))
            results[name] = res
        return results

    def benchmark(self, suite: list[GraphPair] | None = None) -> PipelineReport:
        suite = suite or make_benchmark_suite(self.cfg)
        rows: list[BenchmarkRow] = []
        for gp in suite:
            results = self.run_pair(gp)
            for name, res in results.items():
                rec = recovery_rate(res.perm, gp.true_perm) if gp.true_perm is not None else 0.0
                rows.append(
                    BenchmarkRow(
                        method=name,
                        noise=gp.noise,
                        seed=int(gp.seed),
                        n=gp.n,
                        recovery=rec,
                        score=res.score,
                        distortion=res.distortion,
                        elapsed_sec=res.meta.get("elapsed_sec", 0.0),
                    )
                )
        report = PipelineReport(rows=rows)
        report.aggregated = self._aggregate(rows)
        return report

    @staticmethod
    def _aggregate(rows: list[BenchmarkRow]) -> dict[str, dict[float, dict[str, float]]]:
        agg: dict[str, dict[float, dict[str, float]]] = {}
        for r in rows:
            agg.setdefault(r.method, {}).setdefault(
                r.noise, {"rec": [], "dist": [], "score": [], "t": []}
            )
            agg[r.method][r.noise]["rec"].append(r.recovery)
            agg[r.method][r.noise]["dist"].append(r.distortion)
            agg[r.method][r.noise]["score"].append(r.score)
            agg[r.method][r.noise]["t"].append(r.elapsed_sec)
        for m in agg:
            for nz in agg[m]:
                for k in ("rec", "dist", "score", "t"):
                    v = np.array(agg[m][nz][k], dtype=float)
                    agg[m][nz][k] = {
                        "mean": float(v.mean()),
                        "std": float(v.std()),
                        "min": float(v.min()),
                        "max": float(v.max()),
                    }
        return agg
