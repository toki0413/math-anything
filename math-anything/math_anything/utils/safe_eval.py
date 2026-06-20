"""安全的表达式求值引擎.

替代 eval()，使用 AST 解析实现受限的表达式求值。
仅支持：比较运算、算术运算、布尔运算、属性访问、函数调用（白名单）。
"""

from __future__ import annotations

import ast
import operator
from typing import Any

# 支持的比较运算符
_COMPARE_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

# 支持的算术运算符
_ARITH_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# 支持的布尔运算符
_BOOL_OPS = {
    ast.And: all,
    ast.Or: any,
}


class SafeEvalError(Exception):
    """安全求值错误."""

    pass


# 危险模式黑名单，用于 eval() 调用前的输入验证
_DANGEROUS_PATTERNS = (
    "__",  # dunder 访问
    "import",  # 导入语句
    "exec",  # 代码执行
    "compile",  # 编译调用
    "open",  # 文件操作
    "eval",  # 嵌套 eval
    "globals",  # 全局变量访问
    "locals",  # 局部变量访问
    "vars",  # 变量访问
    "dir",  # 属性枚举
    "getattr",  # 动态属性访问 (eval 上下文中)
    "setattr",  # 动态属性设置
    "delattr",  # 动态属性删除
    "type(",  # 动态类型创建
    "class ",  # 类定义
    "def ",  # 函数定义 (lambda 除外)
    "raise",  # 异常抛出
    "yield",  # 生成器
    "await",  # 异步
    "assert",  # 断言
    "breakpoint",  # 调试
    "exit",  # 退出
    "quit",  # 退出
)


def validate_eval_expr(expr: str) -> None:
    """验证表达式字符串是否包含危险模式.

    用于 eval() 调用前的输入消毒。当 safe_eval 无法替代 eval 时
    (如 numpy 向量化运算)，至少确保表达式不包含明显的危险操作。

    Args:
        expr: 要验证的表达式字符串

    Raises:
        ValueError: 表达式包含潜在危险的模式
    """
    if not isinstance(expr, str):
        raise ValueError(f"Expression must be str, got {type(expr).__name__}")

    if len(expr) > 10000:
        raise ValueError("Expression too long (max 10000 chars)")

    expr_lower = expr.lower()
    for pattern in _DANGEROUS_PATTERNS:
        if pattern in expr_lower:
            # 允许 lambda 表达式 (optimized_evaluator 等模块需要)
            if pattern == "def " and "lambda" in expr_lower:
                continue
            raise ValueError(f"Expression contains potentially dangerous pattern: '{pattern}'")


def safe_eval(expr: str, context: dict[str, Any] | None = None) -> Any:
    """安全地求值表达式.

    与 eval() 不同，此函数：
    1. 不支持赋值、导入、类定义等语句
    2. 不支持 __dunder__ 属性访问
    3. 仅支持白名单内的运算符和函数
    4. 不支持任意函数调用

    Args:
        expr: 要求值的表达式字符串
        context: 变量上下文（替代 eval 的 locals）

    Returns:
        表达式的求值结果

    Raises:
        SafeEvalError: 表达式包含不安全的操作
    """
    if context is None:
        context = {}

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise SafeEvalError(f"Syntax error: {e}") from e

    return _eval_node(tree.body, context)


def _eval_node(node: ast.AST, context: dict[str, Any]) -> Any:
    """递归求值 AST 节点."""

    # 常量
    if isinstance(node, ast.Constant):
        return node.value

    # 名称查找
    if isinstance(node, ast.Name):
        if node.id.startswith("__") and node.id.endswith("__"):
            raise SafeEvalError(f"Dunder access not allowed: {node.id}")
        if node.id in context:
            return context[node.id]
        raise SafeEvalError(f"Undefined variable: {node.id}")

    # 比较运算
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, context)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_node(comparator, context)
            op_func = _COMPARE_OPS.get(type(op))
            if op_func is None:
                raise SafeEvalError(f"Unsupported comparison: {type(op).__name__}")
            if not op_func(left, right):
                return False
            left = right
        return True

    # 布尔运算
    if isinstance(node, ast.BoolOp):
        op_func = _BOOL_OPS.get(type(node.op))
        if op_func is None:
            raise SafeEvalError(f"Unsupported boolean op: {type(node.op).__name__}")
        values = [_eval_node(v, context) for v in node.values]
        return op_func(values)

    # 一元运算
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, context)
        if isinstance(node.op, ast.Not):
            return not operand
        op_func = _ARITH_OPS.get(type(node.op))
        if op_func is None:
            raise SafeEvalError(f"Unsupported unary op: {type(node.op).__name__}")
        return op_func(operand)

    # 二元算术运算
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, context)
        right = _eval_node(node.right, context)
        op_func = _ARITH_OPS.get(type(node.op))
        if op_func is None:
            raise SafeEvalError(f"Unsupported binary op: {type(node.op).__name__}")
        return op_func(left, right)

    # 属性访问
    if isinstance(node, ast.Attribute):
        value = _eval_node(node.value, context)
        attr = node.attr
        if attr.startswith("__") and attr.endswith("__"):
            raise SafeEvalError(f"Dunder access not allowed: {attr}")
        try:
            return getattr(value, attr)
        except AttributeError:
            raise SafeEvalError(f"No attribute '{attr}' on {type(value).__name__}")

    # 字典/对象下标访问
    if isinstance(node, ast.Subscript):
        value = _eval_node(node.value, context)
        key = _eval_node(node.slice, context)
        try:
            return value[key]
        except (KeyError, IndexError, TypeError):
            raise SafeEvalError(f"Cannot subscript {type(value).__name__} with {key!r}")

    # 条件表达式 (a if cond else b)
    if isinstance(node, ast.IfExp):
        test = _eval_node(node.test, context)
        if test:
            return _eval_node(node.body, context)
        return _eval_node(node.orelse, context)

    # 函数/方法调用 (白名单)
    if isinstance(node, ast.Call):
        # 允许特定内置函数调用
        if isinstance(node.func, ast.Name):
            allowed_builtins = {
                "max",
                "min",
                "abs",
                "len",
                "round",
                "sum",
                "int",
                "float",
                "str",
                "bool",
                "list",
                "dict",
                "tuple",
                "set",
                "sorted",
                "enumerate",
                "zip",
                "range",
                "isinstance",
                "type",
                "hasattr",
                "getattr",
            }
            if node.func.id in allowed_builtins:
                func = {
                    "max": max,
                    "min": min,
                    "abs": abs,
                    "len": len,
                    "round": round,
                    "sum": sum,
                    "int": int,
                    "float": float,
                    "str": str,
                    "bool": bool,
                    "list": list,
                    "dict": dict,
                    "tuple": tuple,
                    "set": set,
                    "sorted": sorted,
                    "enumerate": enumerate,
                    "zip": zip,
                    "range": range,
                    "isinstance": isinstance,
                    "type": type,
                    "hasattr": hasattr,
                    "getattr": getattr,
                }[node.func.id]
                args = [_eval_node(a, context) for a in node.args]
                kwargs = {kw.arg: _eval_node(kw.value, context) for kw in node.keywords if kw.arg}
                return func(*args, **kwargs)
            # 也检查 context 中的函数
            if node.func.id in context and callable(context[node.func.id]):
                args = [_eval_node(a, context) for a in node.args]
                kwargs = {kw.arg: _eval_node(kw.value, context) for kw in node.keywords if kw.arg}
                return context[node.func.id](*args, **kwargs)
            raise SafeEvalError(f"Function call not allowed: {node.func.id}")
        # 允许特定方法调用
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            allowed_methods = {
                "lower",
                "upper",
                "strip",
                "lstrip",
                "rstrip",
                "split",
                "join",
                "replace",
                "startswith",
                "endswith",
                "format",
                "isdigit",
                "isalpha",
                "isnumeric",
                "get",
                "items",
                "keys",
                "values",
                "append",
                "extend",
                "pop",
                # numpy/math 常用函数
                "sin",
                "cos",
                "tan",
                "arcsin",
                "arccos",
                "arctan",
                "exp",
                "log",
                "log2",
                "log10",
                "sqrt",
                "abs",
                "floor",
                "ceil",
                "round",
                "sign",
                "sum",
                "min",
                "max",
                "mean",
                "std",
                "var",
                "clip",
                "where",
                "reshape",
                "T",
                "shape",
                "pi",
                "e",
            }
            if method_name not in allowed_methods:
                raise SafeEvalError(f"Method call not allowed: {method_name}")
            obj = _eval_node(node.func.value, context)
            args = [_eval_node(a, context) for a in node.args]
            kwargs = {kw.arg: _eval_node(kw.value, context) for kw in node.keywords if kw.arg}
            method = getattr(obj, method_name)
            return method(*args, **kwargs)
        raise SafeEvalError("Function calls not allowed (except whitelisted methods)")

    # 列表/元组字面量
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_node(e, context) for e in node.elts]

    # 字典字面量
    if isinstance(node, ast.Dict):
        return {_eval_node(k, context): _eval_node(v, context) for k, v in zip(node.keys, node.values)}

    # Set 字面量
    if isinstance(node, ast.Set):
        return {_eval_node(e, context) for e in node.elts}

    raise SafeEvalError(f"Unsupported AST node: {type(node).__name__}")
