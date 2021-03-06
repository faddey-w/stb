import math
import random
from stb.ai.evolution.lib import ComputeGraph, atan2, asin, sin, cos


def test_identity():
    g = ComputeGraph("x")
    assert _is_close(g["x"].eval({"x": 1.123}), 1.123)
    assert _is_close(g["x"].eval({"x": -1.987}), -1.987)
    assert len(g.expressions) == 1


def test_add_exprs():
    g = ComputeGraph("xyz")
    assert _is_close((g["x"] + g["y"] + g["z"]).eval(dict(x=1, y=20, z=300)), 321.0)
    assert len(g.expressions) == 5


def test_add_with_const():
    g = ComputeGraph("x")
    assert _is_close((g["x"] + 5).eval({"x": 1.5}), 6.5)
    assert len(g.expressions) == 2


def test_radd():
    g = ComputeGraph("x")
    assert _is_close((7 + g["x"]).eval({"x": -4.25}), 2.75)
    assert len(g.expressions) == 2


def test_sub():
    g = ComputeGraph("x")
    assert _is_close((g["x"] - 5.7).eval({"x": 1.5}), -4.2)
    assert len(g.expressions) == 2


def test_rsub():
    g = ComputeGraph("x")
    assert _is_close((8 - g["x"]).eval({"x": -4.25}), 12.25)
    assert len(g.expressions) == 2


def test_neg():
    g = ComputeGraph("x")
    assert _is_close((-g["x"]).eval({"x": -4.25}), 4.25)
    assert len(g.expressions) == 2


def test_mul():
    g = ComputeGraph("xy")
    assert _is_close((g["x"] * g["y"] * g["x"]).eval({"x": -4, "y": 0.3}), 4.8)
    assert len(g.expressions) == 4


def test_rmul():
    g = ComputeGraph("x")
    assert _is_close((3.1 * g["x"]).eval({"x": -4.25}), 3.1 * -4.25)
    assert len(g.expressions) == 2


def test_truediv():
    g = ComputeGraph("xy")
    assert _is_close((g["x"] / g["y"]).eval({"x": -4.3, "y": 0.25}), -4.3 / 0.25)
    assert len(g.expressions) == 3


def test_rtruediv():
    g = ComputeGraph("x")
    assert _is_close((3.1 / g["x"]).eval({"x": -4.25}), 3.1 / -4.25)
    assert len(g.expressions) == 2


def test_truediv_by_zero():
    g = ComputeGraph("xy")
    assert (g["x"] / g["y"]).eval({"x": -4.3, "y": 0.0}) == float("-inf")
    assert len(g.expressions) == 3


def test_pow():
    g = ComputeGraph("x")
    assert _is_close((g["x"] ** 3).eval({"x": 5}), 125.0)
    assert _is_close((g["x"] ** 3).eval({"x": -5}), -125.0)
    assert _is_close((g["x"] ** 1.2).eval({"x": 5}), (5 ** 1.2))
    assert math.isnan((g["x"] ** 1.2).eval({"x": -5}))
    assert len(g.expressions) == 5


def test_floordiv():
    g = ComputeGraph("x")
    assert _is_close((g["x"] // 3.1).eval({"x": 6.35}), 2.0)
    assert _is_close((g["x"] // 3.1).eval({"x": -6.35}), -2.0)
    assert len(g.expressions) == 3


def test_mod():
    g = ComputeGraph("xy")
    assert _is_close((g["x"] % g["y"]).eval({"x": 6.35, "y": 3.1}), 0.15)
    assert len(g.expressions) == 3


def test_abs():
    g = ComputeGraph("x")
    assert _is_close(abs(g["x"]).eval({"x": -55}), 55.0)
    assert len(g.expressions) == 2


def test_ge():
    g = ComputeGraph("xy")
    assert _is_close((g["x"] >= g["y"]).eval({"x": 6.3, "y": 3.1}), 1.0)
    assert _is_close((g["x"] >= g["y"]).eval({"x": 2.1, "y": 3.1}), 0.0)
    assert _is_close((g["x"] >= g["y"]).eval({"x": 3.1, "y": 3.1}), 1.0)
    assert len(g.expressions) == 5


def test_le():
    g = ComputeGraph("xy")
    assert _is_close((g["x"] <= g["y"]).eval({"x": 6.3, "y": 3.1}), 0.0)
    assert _is_close((g["x"] <= g["y"]).eval({"x": 2.1, "y": 3.1}), 1.0)
    assert _is_close((g["x"] <= g["y"]).eval({"x": 3.1, "y": 3.1}), 1.0)
    assert len(g.expressions) == 5


def test_gt():
    g = ComputeGraph("xy")
    assert _is_close((g["x"] > g["y"]).eval({"x": 6.3, "y": 3.1}), 1.0)
    assert _is_close((g["x"] > g["y"]).eval({"x": 2.1, "y": 3.1}), 0.0)
    assert _is_close((g["x"] > g["y"]).eval({"x": 3.1, "y": 3.1}), 0.0)
    assert len(g.expressions) == 5


def test_lt():
    g = ComputeGraph("xy")
    assert _is_close((g["x"] < g["y"]).eval({"x": 6.3, "y": 3.1}), 0.0)
    assert _is_close((g["x"] < g["y"]).eval({"x": 2.1, "y": 3.1}), 1.0)
    assert _is_close((g["x"] < g["y"]).eval({"x": 3.1, "y": 3.1}), 0.0)
    assert len(g.expressions) == 5


def test_atan2():
    g = ComputeGraph("xy")
    z = atan2(g["y"], g["x"])

    n = 72
    cases = []
    for i in range(n):
        angle = math.pi * 2 * i / n
        radius = random.uniform(0.1, 1000)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        cases.append((y, x))
    _assert_approx_function(z, math.atan2, cases, arg_names="yx")
    # assert math.isnan(z.eval({"x": 0, "y": 0}))  # don't care about this case


def test_asin():
    g = ComputeGraph("x")
    y = asin(g["x"])

    n = 10
    _assert_approx_function(y, math.asin, [2.0 * i / n - 1 for i in range(n + 1)])
    assert math.isnan(y.eval({"x": -1.1}))
    assert math.isnan(y.eval({"x": +1.1}))


def test_sin():
    g = ComputeGraph("x")
    y = sin(g["x"])

    n = 10
    _assert_approx_function(
        y, math.sin, [i * 2 * math.pi / n for i in range(-2 * n, 2 * n)]
    )


def test_cos():
    g = ComputeGraph("x")
    y = cos(g["x"])

    n = 10
    _assert_approx_function(
        y, math.cos, [i * 2 * math.pi / n for i in range(-2 * n, 2 * n)]
    )


def _assert_approx_function(
    value_expr, true_fn, arg_cases, abs_err=0.0005, rel_err=0.0005, arg_names="x"
):
    for args in arg_cases:
        if not isinstance(args, tuple):
            args = (args,)
        assert len(args) <= len(arg_names)
        args_dict = dict(zip(arg_names, args))
        assert _is_close(
            value_expr.eval(args_dict), true_fn(*args), abs_err=abs_err, rel_err=rel_err
        )


def _is_close(x1, x2, abs_err=1.0e-7, rel_err=1.0e-5):
    err = x1 - x2
    if -abs_err <= err <= abs_err:
        return True
    if (x1 < 0) ^ (x2 < 0):
        return False
    x1_abs = abs(x1)
    x2_abs = abs(x2)
    if x1_abs > x2_abs:
        rel = x2_abs / x1_abs
    else:
        rel = x1_abs / x2_abs
    return (1 - rel) <= rel_err
