# MatchForge · core/errors.py
# 错误码目录（E100~E500），所有异常继承自 MatchForgeError。作者：晨星
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorSpec:
    code: str
    message: str
    hint: str = ""


_ERROR_CATALOG: dict[str, ErrorSpec] = {
    "E100": ErrorSpec("E100", "Graph data invalid", "A/B 必须是对称 0/1 邻接矩阵且形状 (n,n)"),
    "E101": ErrorSpec("E101", "Permutation length mismatch", "perm 长度必须等于图节点数 n"),
    "E102": ErrorSpec(
        "E102", "Non-permutation matrix", "P 必须是 (0/1) 置换矩阵，每行每列恰有一个 1"
    ),
    "E200": ErrorSpec(
        "E200", "Matcher not available", "所需后端不可用（依赖缺失），已降级到离线兜底"
    ),
    "E201": ErrorSpec(
        "E201", "FAQ did not converge", "Frank-Wolfe 未达相对收敛阈值，可能为病态输入"
    ),
    "E300": ErrorSpec("E300", "Pipeline configuration error", "benchmark suite 构造不合法"),
    "E301": ErrorSpec("E301", "Benchmark suite empty", "make_benchmark_suite 返回空列表"),
    "E400": ErrorSpec("E400", "Config value error", "ENV 覆盖值类型/取值非法"),
    "E500": ErrorSpec(
        "E500", "Numerical invariant violated", "数学不变量校验失败，实现或被测假设有误"
    ),
}


class MatchForgeError(Exception):
    def __init__(self, code: str, *args: object) -> None:
        spec = _ERROR_CATALOG.get(code)
        if spec is None:
            super().__init__(f"Unknown error code {code}")
            self.code = code
            return
        msg = spec.message
        if args:
            msg = msg + " -> " + " ".join(str(a) for a in args)
        super().__init__(f"[{code}] {msg}")
        self.code = code
        self.hint = spec.hint

    @property
    def catalog(self) -> ErrorSpec | None:
        return _ERROR_CATALOG.get(self.code)


def error_class(code: str):
    """按错误码生成专用异常类（重码/重名即 RuntimeError，见 pitfalls G 节）。"""
    spec = _ERROR_CATALOG.get(code)
    if spec is None:
        raise MatchForgeError("E400", f"undefined code {code}")
    if code in _REGISTRY:
        raise RuntimeError(f"error code already registered: {code}")
    _REGISTRY[code] = True

    class _E(MatchForgeError):
        def __init__(self, *args: object) -> None:
            super().__init__(code, *args)

    _E.__name__ = code
    return _E


_REGISTRY: dict[str, bool] = {}


# 专用异常类（构造即注册，重名/重码会在导入期炸）
GraphDataError = error_class("E100")
PermutationError = error_class("E101")
NonPermutationMatrixError = error_class("E102")
MatcherUnavailableError = error_class("E200")
FAQNotConvergedError = error_class("E201")
PipelineConfigError = error_class("E300")
BenchmarkEmptyError = error_class("E301")
ConfigValueError = error_class("E400")
InvariantError = error_class("E500")
