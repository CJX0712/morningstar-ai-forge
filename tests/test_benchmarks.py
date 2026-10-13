"""L10 集成验证：在所有模块上跑「不变量 + 跨库」闭环，必须全绿。"""
import pytest

from morningstar_ai.benchmarks.run import run_all


def test_all_modules_pass():
    rep = run_all(verbose=False)
    assert rep["all_pass"], rep
    assert rep["invariants_ok"] == rep["total"]
    assert rep["crosslib_ok"] == rep["crosslib_total"]
