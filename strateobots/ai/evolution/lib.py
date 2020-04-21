import contextlib
import math
from collections import namedtuple
from strateobots.ai.evolution import evo_core


class ComputeGraph:
    def __init__(self, arg_names):
        self._arg_names = list(arg_names)
        self.expressions = []
        self.register_enabled = True
        for i in range(len(self._arg_names)):
            Expression(self, param_i=i)

    def register(self, expression: "Expression"):
        if self.register_enabled:
            expression.id = len(self.expressions)
            self.expressions.append(expression)

    def __getitem__(self, item) -> "Expression":
        if isinstance(item, str):
            i = self._arg_names.index(item)
        elif isinstance(item, int):
            i = item
        else:
            raise TypeError(type(item))
        return self.expressions[i]

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
        if not (isinstance(args, dict) and set(args) == set(self._arg_names)):
            raise ValueError(
                "Invalid arguments, expected dict with keys: "
                + ", ".join(self._arg_names)
            )
        if not expr_ids:
            return []
        max_expr_id = max(expr_ids)
        result_values = [args[arg_name] for arg_name in self._arg_names]
        for i in range(len(result_values), max_expr_id + 1):
            value = self.expressions[i].value.evaluate(result_values)
            result_values.append(value)
        return [result_values[i] for i in expr_ids]

    @contextlib.contextmanager
    def register_mode(self, enabled):
        orig = self.register_enabled
        self.register_enabled = enabled
        yield
        self.register_enabled = orig


class Expression:
    def __init__(self, graph: ComputeGraph, *, expr=None, param_i=None, constant=None):
        assert (expr, param_i, constant).count(None) == 2
        self.graph = graph
        self.id = None
        graph.register(self)

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
        value = self / other
        value.value.set_special_function(evo_core.SpecialFunction.Truncate)
        return value

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
        value = self - other
        value.value.set_special_function(evo_core.SpecialFunction.Threshold)
        return value

    def __le__(self, other):
        value = other - self
        value.value.set_special_function(evo_core.SpecialFunction.Threshold)
        return value

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
    with tan.graph.register_mode(False):
        tan_sqr_plus_1 = 1 + tan ** 2
    result = _expr_with_args(tan)
    result.add_child(tan_sqr_plus_1.value)
    # Euler series which converges faster than Taylor:
    coef = 1.0
    for n in range(20):
        if n > 0:
            coef *= 4.0 * n * n / ((2 * n) * (2 * n + 1))
        result.add_poly_member(coef, [0, 1], [2 * n + 1, -n - 1])
    return Expression(tan.graph, expr=result)


def atan2(dy: Expression, dx: Expression) -> Expression:
    graph = dy.graph
    tan = dy / dx
    with graph.register_mode(False):
        dx_is_zero = abs(dx) <= 0
    dx_is_zero = Expression(graph, expr=dx_is_zero.value)
    with graph.register_mode(False):
        dy_sign = (2 * (dy > 0)) - 1
        dx_non_zero = 1 - dx_is_zero
        dx_negative = dx < 0
        result = dx_non_zero * atan(tan)
        result += dy_sign * (dx_negative * math.pi + dx_is_zero * (math.pi / 2))

    return Expression(graph, expr=result.value)
