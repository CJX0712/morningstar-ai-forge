"""测试共享工具：递归提取各模块 check_invariants 的 passed 字段。

各模块返回结构不统一（有的顶层 passed，有的嵌套），统一递归收集所有
passed 布尔并取与，等价于 benchmarks/run._extract_passed 的世界级判定。
"""


def all_passed(d):
    found = []

    def _walk(o):
        if isinstance(o, dict):
            if "passed" in o:
                found.append(bool(o["passed"]))
            for v in o.values():
                _walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                _walk(v)

    _walk(d)
    return all(found) if found else True
