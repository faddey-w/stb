import contextlib
import math
from collections import namedtuple
from stb.ai.evolution import evo_core


class ComputeGraph:
    def __init__(self, arg_names):
        self.arg_names = tuple(arg_names)
        self.expressions = []
        self.graph = evo_core.Graph()
        self.register_enabled = True
        for i in range(len(self.arg_names)):
            Expression(self, param_i=i)

    def register(self, expression: "Expression"):
        if self.register_enabled:
            expression.id = len(self.expressions)
            self.expressions.append(expression)
            if expression.id >= len(self.arg_names):
                self.graph.add_expr(expression.value)

    def __getitem__(self, item) -> "Expression":
        if isinstance(item, str):
            i = self.arg_names.index(item)
        elif isinstance(item, int):
            i = item
        else:
            raise TypeError(type(item))
        return self.expressions[i]

    def const(self, value):
        return Expression(self, constant=value)

    def eval(self, args, expressions_or_ids):
        expr_ids = []
        for expr in expressions_or_ids:
            if isinstance(expr, Expression):
                if expr.graph is not self:
                    raise ValueError("Expression from another graph")
                expr_ids.append(expr.id)
            elif isinstance(expr, int):
                if not (0 <= expr < len(self.expressions)):
                    raise ValueError("Invalid expression id")
                expr_ids.append(expr)
            else:
                raise TypeError("Unrecognized expression type: " + str(type(expr)))
        if not (isinstance(args, dict) and set(args) == set(self.arg_names)):
            raise ValueError(
                "Invalid arguments, expected dict with keys: "
                + ", ".join(self.arg_names)
            )
        result = [0.0] * len(expr_ids)
        id_map = []
        n_args = len(self.arg_names)
        param_values = [args[arg_name] for arg_name in self.arg_names]
        for i, expr_id in enumerate(expr_ids):
            if expr_id < n_args:
                result[i] = param_values[expr_id]
            else:
                id_map.append((expr_id, i))
        if not id_map:
            return result
        values = self.graph.evaluate(param_values)
        for expr_id, res_i in id_map:
            result[res_i] = values[expr_id - n_args]
        return result

    @contextlib.contextmanager
    def register_mode(self, enabled):
        orig = self.register_enabled
        self.register_enabled = enabled
        yield
        self.register_enabled = orig

    def __repr__(self):
        def name_fn(i):
            if i < n_args:
                return self.arg_names[i].replace("/", ":")
            else:
                return f"_{i-n_args+1}"

        n_args = len(self.arg_names)
        return self.graph.to_str(n_args, name_fn)

    def get_out_index(self, expr):
        return expr.id - len(self.arg_names)


class Expression:
    def __init__(self, graph: ComputeGraph, *, expr=None, param_i=None, constant=None):
        assert (expr, param_i, constant).count(None) == 2
        self.graph = graph
        self.id = None

        if expr is not None:
            self.value = expr
        elif param_i is not None:
            value = evo_core.TreeExpression()
            value.set_linear([param_i], [1.0], 0.0)
            self.value = value
        elif constant is not None:
            value = evo_core.TreeExpression()
            value.set_linear([], [], constant)
            self.value = value
        graph.register(self)

    def eval(self, args):
        return self.graph.eval(args, [self])[0]

    def __add__(self, other):
        value = _linear([self, other], [1, 1])
        return Expression(self.graph, expr=value)

    def __radd__(self, other):
        value = _linear([other, self], [1, 1])
        return Expression(self.graph, expr=value)

    def __sub__(self, other):
        value = _linear([self, other], [1, -1])
        return Expression(self.graph, expr=value)

    def __rsub__(self, other):
        value = _linear([other, self], [1, -1])
        return Expression(self.graph, expr=value)

    def __neg__(self):
        value = _linear([self], [-1])
        return Expression(self.graph, expr=value)

    def __mul__(self, other):
        value = _product([self, other], [1, 1])
        return Expression(self.graph, expr=value)

    def __rmul__(self, other):
        value = _product([other, self], [1, 1])
        return Expression(self.graph, expr=value)

    def __truediv__(self, other):
        value = _product([self, other], [1, -1])
        return Expression(self.graph, expr=value)

    def __rtruediv__(self, other):
        value = _product([other, self], [1, -1])
        return Expression(self.graph, expr=value)

    def __pow__(self, power, modulo=None):
        if modulo is not None:
            raise NotImplementedError
        if not isinstance(power, (int, float)):
            raise TypeError
        value = _product([self], [power])
        return Expression(self.graph, expr=value)

    def __floordiv__(self, other):
        with self.graph.register_mode(False):
            value = self / other
            value.value.set_special_function(evo_core.SpecialFunction.Truncate)
        return Expression(self.graph, expr=value.value)

    def __mod__(self, other):
        with self.graph.register_mode(False):
            ratio = self / other
            ratio.value.set_special_function(evo_core.SpecialFunction.Remainder)
            result = other * ratio
        return Expression(self.graph, expr=result.value)

    def __abs__(self):
        is_positive = _linear([self], [1])
        is_positive.set_special_function(evo_core.SpecialFunction.Threshold)
        multiplier = _linear([is_positive, 1], [2, -1])
        result = _product([multiplier, self], [1, 1])
        return Expression(self.graph, expr=result)

    def __ge__(self, other):
        with self.graph.register_mode(False):
            value = self - other
            value.value.set_special_function(evo_core.SpecialFunction.Threshold)
        return Expression(self.graph, expr=value.value)

    def __le__(self, other):
        with self.graph.register_mode(False):
            value = other - self
            value.value.set_special_function(evo_core.SpecialFunction.Threshold)
        return Expression(self.graph, expr=value.value)

    def __gt__(self, other):
        with self.graph.register_mode(False):
            ret = self <= other
            ret = 1 - ret
        return Expression(self.graph, expr=ret.value)

    def __lt__(self, other):
        with self.graph.register_mode(False):
            ret = self >= other
            ret = 1 - ret
        return Expression(self.graph, expr=ret.value)


_ExprType = namedtuple("_ExprType", "is_const is_raw is_registered")


def _check_expr(thing) -> _ExprType:
    if isinstance(thing, (int, float)):
        return _ExprType(is_const=True, is_raw=False, is_registered=False)
    if isinstance(thing, evo_core.TreeExpression):
        return _ExprType(is_const=False, is_raw=True, is_registered=False)
    if not isinstance(thing, Expression):
        raise TypeError(type(thing))
    return _ExprType(is_const=False, is_raw=False, is_registered=thing.id is not None)


def _expr_with_args(*args):
    value = evo_core.TreeExpression()
    for arg in args:
        expr_t = _check_expr(arg)
        if expr_t.is_const:
            arg_expr = evo_core.TreeExpression()
            arg_expr.set_linear([], [], arg)
        elif expr_t.is_raw:
            arg_expr = arg
        elif expr_t.is_registered:
            arg_expr = evo_core.TreeExpression()
            arg_expr.set_linear([arg.id], [1.0], 0.0)
        else:
            arg_expr = arg.value
        value.add_child(arg_expr)
    return value


def _linear(args, coefs):
    result = evo_core.TreeExpression()
    linear_ids = []
    linear_coefs = []
    linear_bias = 0
    n_childs = 0
    for arg, coef in zip(args, coefs):
        arg_t = _check_expr(arg)
        if arg_t.is_const:
            linear_bias += arg * coef
        elif arg_t.is_registered:
            linear_ids.append(arg.id)
            linear_coefs.append(coef)
        else:
            result.add_child(arg if arg_t.is_raw else arg.value)
            result.add_poly_member(coef, [n_childs], [1])
            n_childs += 1
    result.set_linear(linear_ids, linear_coefs, linear_bias)
    return result


def _product(args, powers, coef=1):
    childs = []
    child_powers = []
    for arg, power in zip(args, powers):
        arg_t = _check_expr(arg)
        if arg_t.is_const:
            coef *= arg ** power
        else:
            childs.append(arg)
            child_powers.append(power)
    result = _expr_with_args(*childs)
    result.add_poly_member(coef, list(range(len(childs))), child_powers)
    return result


def atan(tan: Expression) -> Expression:
    return _atan(tan)


def _atan(tan: Expression, coef=1.0) -> Expression:
    with tan.graph.register_mode(False):
        tan_sqr_plus_1 = 1 + tan ** 2
    result = _expr_with_args(tan)
    result.add_child(tan_sqr_plus_1.value)
    # Euler series which converges faster than Taylor:
    for n in range(10):
        if n > 0:
            coef *= 4.0 * n * n / ((2 * n) * (2 * n + 1))
        result.add_poly_member(coef, [0, 1], [2 * n + 1, -n - 1])
    return Expression(tan.graph, expr=result)


def atan2(dy: Expression, dx: Expression) -> Expression:
    graph = dy.graph
    tan = dy / dx
    ctg = dx / dy
    dx_is_negative = dx <= 0
    dy_is_negative = dy <= 0
    with graph.register_mode(False):
        use_tan = (dx * (1 - 2 * dx_is_negative)) >= (dy * (1 - 2 * dy_is_negative))
    use_tan = Expression(graph, expr=use_tan.value)
    with graph.register_mode(False):
        angle1 = _atan(tan) + dx_is_negative * math.pi - 2 * math.pi * dx_is_negative * dy_is_negative
        angle2 = (math.pi / 2) - _atan(ctg) - math.pi * dy_is_negative
        result = use_tan * angle1 + (1 - use_tan) * angle2

    return Expression(graph, expr=result.value)


def sin(x: Expression) -> Expression:
    with x.graph.register_mode(False):
        x = ((x + math.pi) % (2 * math.pi)) - math.pi
    x = Expression(x.graph, expr=x.value)
    result = _expr_with_args(x)
    coef = 1
    for n in range(1, 7):
        result.add_poly_member(1 / coef, [0], [2 * n - 1])
        coef *= -(2 * n) * (2 * n + 1)
    return Expression(x.graph, expr=result)


def cos(x: Expression) -> Expression:
    with x.graph.register_mode(False):
        x = ((x + math.pi) % (2 * math.pi)) - math.pi
    x = Expression(x.graph, expr=x.value)
    result = _expr_with_args(x)
    result.set_linear([], [], 1)
    coef = 1
    for n in range(1, 7):
        coef *= -(2 * n - 1) * (2 * n)
        result.add_poly_member(1.0 / coef, [0], [2 * n])
    return Expression(x.graph, expr=result)


def asin(x: Expression) -> Expression:
    with x.graph.register_mode(False):
        tan_half = x / (1 + (1 - x ** 2) ** 0.5)
    tan_half = Expression(x.graph, expr=tan_half.value)
    return _atan(tan_half, 2)


def logical_or(x, y):
    with x.graph.register_mode(False):
        z = 1 - (1 - x) * (1 - y)
    return Expression(x.graph, expr=z.value)


def binary_choice(predicate, true_value, false_value):
    with predicate.graph.register_mode(False):
        result = predicate * true_value + (1 - predicate) * false_value
    return Expression(predicate.graph, expr=result.value)
