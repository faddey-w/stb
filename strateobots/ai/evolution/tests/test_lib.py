from strateobots.ai.evolution.lib import ComputeGraph


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
    assert (g["x"] / g["y"]).eval({"x": -4.3, "y": 0.}) == float("-inf")
    assert len(g.expressions) == 3


def test_pow():
    g = ComputeGraph("x")
    assert _is_close((g["x"] ** 3).eval({"x": 5}), 125.0)
    assert len(g.expressions) == 2


def test_floordiv():
    g = ComputeGraph("x")
    assert _is_close((g["x"] // 3.1).eval({"x": 6.35}), 2.0)
    assert _is_close((g["x"] // 3.1).eval({"x": -6.35}), -2.0)
    assert len(g.expressions) == 3


def test_mod():
    g = ComputeGraph("xy")
    assert _is_close((g["x"] % g['y']).eval({"x": 6.35, 'y': 3.1}), 0.15)
    assert len(g.expressions) == 3


def test_abs():
    g = ComputeGraph("x")
    assert _is_close(abs(g["x"]).eval({"x": -55}), 55.0)
    assert len(g.expressions) == 2


def test_ge():
    g = ComputeGraph("xy")
    assert _is_close((g["x"] >= g['y']).eval({"x": 6.3, 'y': 3.1}), 1.0)
    assert _is_close((g["x"] >= g['y']).eval({"x": 2.1, 'y': 3.1}), 0.0)
    assert _is_close((g["x"] >= g['y']).eval({"x": 3.1, 'y': 3.1}), 1.0)
    assert len(g.expressions) == 5


def test_le():
    g = ComputeGraph("xy")
    assert _is_close((g["x"] <= g['y']).eval({"x": 6.3, 'y': 3.1}), 0.0)
    assert _is_close((g["x"] <= g['y']).eval({"x": 2.1, 'y': 3.1}), 1.0)
    assert _is_close((g["x"] <= g['y']).eval({"x": 3.1, 'y': 3.1}), 1.0)
    assert len(g.expressions) == 5


def test_gt():
    g = ComputeGraph("xy")
    assert _is_close((g["x"] > g['y']).eval({"x": 6.3, 'y': 3.1}), 1.0)
    assert _is_close((g["x"] > g['y']).eval({"x": 2.1, 'y': 3.1}), 0.0)
    assert _is_close((g["x"] > g['y']).eval({"x": 3.1, 'y': 3.1}), 0.0)
    assert len(g.expressions) == 5


def test_lt():
    g = ComputeGraph("xy")
    assert _is_close((g["x"] < g['y']).eval({"x": 6.3, 'y': 3.1}), 0.0)
    assert _is_close((g["x"] < g['y']).eval({"x": 2.1, 'y': 3.1}), 1.0)
    assert _is_close((g["x"] < g['y']).eval({"x": 3.1, 'y': 3.1}), 0.0)
    assert len(g.expressions) == 5


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
