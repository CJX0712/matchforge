# MatchForge 单元测试：CLI 冒烟（MATCHFORGE_* ENV 覆盖 + 门禁落盘）。作者：晨星
from __future__ import annotations

import json

import cli as cli_mod


def test_cli_run_smoke(tmp_path, monkeypatch):
    # 用 ENV 构造极小配置，验证 `run` 子命令可完整跑通门禁并落盘
    monkeypatch.setenv("MATCHFORGE_N_NODES", "30")
    monkeypatch.setenv("MATCHFORGE_SEEDS", "7")
    monkeypatch.setenv("MATCHFORGE_NOISE_LEVELS", "0.0,0.3")
    monkeypatch.setenv("MATCHFORGE_DGP_MODEL", "er")
    monkeypatch.setenv("MATCHFORGE_EDGE_PROB", "0.5")
    monkeypatch.setenv("MATCHFORGE_N_ANCHORS", "5")
    monkeypatch.setenv("MATCHFORGE_FAQ_MAX_ITER", "25")

    out = tmp_path / "benchmark.json"
    rc = cli_mod.main(["run", "--env", "--out", str(out)])
    assert rc == 0
    assert out.exists()

    data = json.loads(out.read_text(encoding="utf-8"))
    g = data["gates"]
    assert g["G1_iso_pass"] is True
    assert g["G2_pass"] is True
    assert g["G3_pass"] is True
