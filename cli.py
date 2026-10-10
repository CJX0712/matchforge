# MatchForge · cli.py
# 命令行入口：run（benchmark + 门禁 + 落盘）。作者：晨星
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.config import Config  # noqa: E402
from examples.run_demo import main as demo_main  # noqa: E402


def cmd_run(args: argparse.Namespace) -> int:
    cfg = Config.from_env() if args.env else Config()
    res = demo_main(cfg)
    out = args.out or os.path.join(ROOT, "examples", "benchmark.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    g = res["gates"]
    print("\n=== Gates ===")
    print(f" G1 iso(η=0) pass : {g['G1_iso_pass']}")
    print(
        f" G2 headline pass : {g['G2_pass']}  (Δ=+{g['G2_delta'] * 100:.1f}pp vs {g['G2_strongest_baseline']})"
    )
    print(f" G3 determinism   : {g['G3_pass']}  (max|Δ|={g['G3_max_abs_diff']:.2e})")
    return 0 if (g["G1_iso_pass"] and g["G2_pass"] and g["G3_pass"]) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="matchforge", description="MatchForge: 世界级图匹配系统 (作者: 晨星)"
    )
    sub = p.add_subparsers(dest="cmd")
    run = sub.add_parser("run", help="运行 benchmark + 门禁 + 落盘")
    run.add_argument("--out", default=None, help="benchmark.json 输出路径")
    run.add_argument("--env", action="store_true", help="从 MATCHFORGE_* 环境变量读取配置")
    run.set_defaults(func=cmd_run)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
