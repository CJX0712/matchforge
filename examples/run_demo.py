# MatchForge · examples/run_demo.py
# 端到端演示：合成图对 → 多算法 benchmark → 门禁 → 落盘 benchmark.json → 确定性二次校验。作者：晨星
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

# 确定性：钉死 BLAS 线程（core.seed 在 import 时已设 os.environ，须在 numpy 前）
from core.seed import set_all

# 确保项目根在 sys.path（CI / 直接运行均可）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.config import Config  # noqa: E402
from data.generators import make_benchmark_suite  # noqa: E402
from pipeline import MatchPipeline  # noqa: E402

AUTHOR = "晨星"
BASELINE_KEYS = ("faq", "grampa", "spectral", "degree_profile", "matchfuse")
# 具备同构恢复能力的方法（random/degree_profile 属下界基线，不参与 G1 同构门禁）
ISO_METHODS = ("spectral", "grampa", "faq", "matchfuse", "seeded")
FLAGSHIP = "seeded"


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def main(cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    set_all(20260610)
    pipe = MatchPipeline(cfg)
    suite = make_benchmark_suite(cfg)

    t0 = time.perf_counter()
    report = pipe.benchmark(suite)
    wall = time.perf_counter() - t0
    agg = report.aggregated

    # ---- 打印恢复率表（method × noise，mean±std）----
    noises = sorted({r.noise for r in report.rows})
    methods = list(agg.keys())
    print("=" * 88)
    print(f" MatchForge · 图匹配 benchmark   n={cfg.n_nodes}  p={cfg.edge_prob}  seeds={cfg.seeds}")
    print("=" * 88)
    header = f" {'method':<14}" + "".join(f" eta={nz:<6}" for nz in noises)
    print(header)
    for m in methods:
        line = f" {m:<14}"
        for nz in noises:
            rec = agg[m][nz]["rec"]
            line += f" {rec['mean'] * 100:5.1f}±{rec['std'] * 100:4.1f}"
        print(line)
    print("-" * 88)

    # ---- 门禁 ----
    # G1 同构恢复率下限（仅考核具备同构恢复能力的方法）
    iso_vals = {m: agg[m][0.0]["rec"]["mean"] for m in ISO_METHODS if m in agg and 0.0 in agg[m]}
    iso_pass = all(v >= cfg.iso_recovery_min for v in iso_vals.values())
    print(
        f" [G1] iso(η=0) recovery ≥ {cfg.iso_recovery_min}: "
        + ("PASS" if iso_pass else "FAIL")
        + "  "
        + ", ".join(f"{m}={iso_vals[m] * 100:.1f}%" for m in iso_vals)
    )

    # G2 头条：旗舰相对最强基线增益
    main = cfg.main_noise
    strongest = max(
        (k for k in BASELINE_KEYS if k in agg), key=lambda k: agg[k][main]["rec"]["mean"]
    )
    fuse_main = agg[FLAGSHIP][main]["rec"]["mean"]
    base_main = agg[strongest][main]["rec"]["mean"]
    delta = fuse_main - base_main
    gate2_pass = delta >= cfg.gate_recovery_gain
    print(
        f" [G2] headline eta={main}: {FLAGSHIP}={fuse_main * 100:.1f}% vs strongest[{strongest}]={base_main * 100:.1f}%"
        f"  Δ=+{delta * 100:.1f}pp  (gate ≥ +{cfg.gate_recovery_gain * 100:.0f}pp) "
        + ("PASS" if gate2_pass else "FAIL")
    )

    # G3 确定性：二次运行
    report2 = MatchPipeline(cfg).benchmark(suite)
    maxdiff = 0.0
    for m in agg:
        for nz in agg[m]:
            maxdiff = max(
                maxdiff, abs(agg[m][nz]["rec"]["mean"] - report2.aggregated[m][nz]["rec"]["mean"])
            )
    det_pass = maxdiff <= 1e-9
    print(f" [G3] determinism max|Δrecovery|={maxdiff:.2e}  " + ("PASS" if det_pass else "FAIL"))

    print(f" [perf] demo wall={wall:.1f}s  (budget ≤ 60s)")

    out = {
        "system": "MatchForge",
        "author": AUTHOR,
        "version": "0.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_nodes": cfg.n_nodes,
            "edge_prob": cfg.edge_prob,
            "noise_levels": list(cfg.noise_levels),
            "seeds": list(cfg.seeds),
            "main_noise": cfg.main_noise,
            "gate_recovery_gain": cfg.gate_recovery_gain,
            "iso_recovery_min": cfg.iso_recovery_min,
        },
        "aggregated": {
            m: {
                str(nz): {
                    "recovery_mean": agg[m][nz]["rec"]["mean"],
                    "recovery_std": agg[m][nz]["rec"]["std"],
                    "distortion_mean": agg[m][nz]["dist"]["mean"],
                    "score_mean": agg[m][nz]["score"]["mean"],
                    "elapsed_mean": agg[m][nz]["t"]["mean"],
                }
                for nz in agg[m]
            }
            for m in agg
        },
        "gates": {
            "G1_iso_recovery_min": cfg.iso_recovery_min,
            "G1_iso_pass": bool(iso_pass),
            "G1_iso_values": iso_vals,
            "G2_main_noise": main,
            "G2_flagship": FLAGSHIP,
            "G2_strongest_baseline": strongest,
            "G2_matchfuse_recovery": fuse_main,
            "G2_baseline_recovery": base_main,
            "G2_delta": delta,
            "G2_gate": cfg.gate_recovery_gain,
            "G2_pass": bool(gate2_pass),
            "G3_max_abs_diff": maxdiff,
            "G3_pass": bool(det_pass),
        },
        "demo_wall_sec": wall,
    }
    return out


if __name__ == "__main__":
    res = main()
    out_path = os.path.join(ROOT, "examples", "benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n benchmark.json written -> {out_path}")
