from .evo_core import LinearExpression, TreeExpression


def test_basic_linear_evaluation():
    le = LinearExpression()
    le.set([0, 1, 2], [13, 19, 23], 0.99)
    assert le.evaluate([10000, 100, 1]) == 131923.99


def test_linear_repr():
    le = LinearExpression()
    le.set([0, 1, 2, 3], [13, -19, -1, 1], 0.99)
    assert str(le) == "13*A-19*B-C+D+0.99"


def test_linear_repr_with_zeros_and_custom_param_names():
    le = LinearExpression()
    le.set([0, 1, 2, 3], [13, 0, -1, 0], 0)
    assert le.to_str("xyzw".__getitem__) == "13*x+0*y-z+0*w"


def test_trivial_tree_evaluation():
    expr = TreeExpression()
    expr.set_linear([0, 1, 2], [13, 19, 23], 0.99)
    assert expr.evaluate([10000, 100, 1]) == 131923.99


def test_trivial_tree_repr():
    expr = TreeExpression()
    expr.set_linear([0, 1, 2], [13, 19, 23], -1.99)
    assert str(expr) == "13*A+19*B+23*C-1.99"


def test_a_bit_complex_tree_expr():
    x = TreeExpression()
    x.set_linear([0], [1], 0)
    two_y_plus_1 = TreeExpression()
    two_y_plus_1.set_linear([1], [2.0], 1)

    expr = TreeExpression()
    expr.add_child(x)
    expr.add_child(two_y_plus_1)
    expr.add_poly_member(1.0, [0], [2])
    expr.add_poly_member(-3.0, [0, 1], [1, 1])
    expr.set_linear([], [], 321)

    assert expr.to_str("xy".__getitem__) == "x^2-3*x*(2*y+1)+321"
    assert expr.evaluate([37, 73]) == (37 ** 2 - 3 * 37 * (2 * 73 + 1) + 321)


def test_tree_construction_semantics():
    x = TreeExpression()
    x.set_linear([0], [1], 0)
    y = TreeExpression()
    y.set_linear([1], [1], 0)

    expr = TreeExpression()

    sub_expr = TreeExpression()
    sub_expr.set_linear([], [], 123)
    expr.add_child(sub_expr)

    sub_expr.set_linear([], [], 321)
    sub_expr.add_child(x)
    sub_expr.add_poly_member(-2, [0], [4])
    expr.add_child(sub_expr)

    sub_expr.add_child(y)
    sub_expr.add_poly_member(1, [0, 1], [1, 1])
    expr.add_child(sub_expr)

    expr.add_poly_member(1, [0, 2], [-1, -1])
    expr.add_poly_member(coefficient=13, child_indices=[1], powers=[5])

    assert str(expr) == "123^-1*(-2*A^4+A*B+321)^-1+13*(-2*A^4+321)^5"
