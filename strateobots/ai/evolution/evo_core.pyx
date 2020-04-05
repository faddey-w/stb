# distutils: language=c++
from libcpp.vector cimport vector
from libc.math cimport pow


cdef struct LinearExpressionNode:
    vector[int] indices
    vector[double] coefficients
    double bias


cdef struct TreeExpressionNode:
    # Let N - amount of env parameters
    #     M - amount of children expressions
    #     P - length of polynom built from children expressions
    #     K_p - amount of factors in p-th polynom member
    vector[TreeExpressionNode] children  # M
    vector[vector[int]] child_indices  # P x K_p
    vector[vector[double]] child_powers  # P x K_p
    vector[double] polynom_coefficients  # P
    LinearExpressionNode linear


cdef double _evaluate_linear_node(LinearExpressionNode* node, vector[double]* params):
    cdef double result = 0.0
    for n in range(node[0].coefficients.size()):
        result += node[0].coefficients[n] * params[0][node[0].indices[n]]
    result += node[0].bias
    return result


cdef double _evaluate_tree_node(TreeExpressionNode* node, vector[double]* params):
    cdef vector[double] values
    cdef double result
    cdef double r

    for m in range(node[0].children.size()):
        values.push_back(_evaluate_tree_node(&node[0].children[m], params))
    result = 0.0
    for p in range(node[0].polynom_coefficients.size()):
        r = 1.0
        for k in range(node[0].child_indices[p].size()):
            r *= pow(values[node[0].child_indices[p][k]], node[0].child_powers[p][k])
        result += r * node[0].polynom_coefficients[p]
    result += _evaluate_linear_node(&node[0].linear, params)
    return result


cdef void _set_linear_node(
        LinearExpressionNode* node,
        vector[int] indices,
        vector[double] coefficients,
        double bias
):
    if indices.size() != coefficients.size():
        raise ValueError("Indices and coefficients should have same number of elements")
    node[0].indices.assign(indices.begin(), indices.end())
    node[0].coefficients.assign(coefficients.begin(), coefficients.end())
    node[0].bias = bias


cdef _add_polynom_member(TreeExpressionNode* node):
    node[0].polynom_coefficients.push_back(0)
    node[0].child_indices.push_back(vector[int]())
    node[0].child_powers.push_back(vector[double]())


cdef _set_polynom_member(
        TreeExpressionNode* node,
        unsigned int member_i,
        double coefficient,
        vector[int] child_indices,
        vector[double] powers,
):
    node[0].polynom_coefficients[member_i] = coefficient
    node[0].child_indices[member_i].assign(child_indices.begin(), child_indices.end())
    node[0].child_powers[member_i].assign(powers.begin(), powers.end())


cdef class LinearExpression:
    cdef LinearExpressionNode data

    cpdef double evaluate(self, vector[double] params):
        return _evaluate_linear_node(&self.data, &params)

    def set(self, indices, coefficients, bias):
        _set_linear_node(&self.data, indices, coefficients, bias)

    def to_str(self, param_name_fn=None, strip_leading_plus=True):
        if param_name_fn is None:
            param_name_fn = _default_param_name_fn
        result = _linear_node_to_str(&self.data, param_name_fn)
        if strip_leading_plus:
            result = result.lstrip("+")
        return result

    def __str__(self):
        return self.to_str()

    def __repr__(self):
        return f"LinearExpression({self.to_str()})"


cdef class TreeExpression:

    cdef TreeExpressionNode data

    cpdef double evaluate(self, vector[double] params):
        return _evaluate_tree_node(&self.data, &params)

    def add_child(self, TreeExpression expr):
        self.data.children.push_back(expr.data)

    def add_poly_member(self, coefficient, child_indices, powers):
        _add_polynom_member(&self.data)
        _set_polynom_member(
            &self.data,
            self.data.child_indices.size()-1,
            coefficient,
            child_indices,
            powers,
        )

    def set_linear(self, indices, coefficients, bias):
        _set_linear_node(&self.data.linear, indices, coefficients, bias)

    def __reduce__(self):
        # Cython can't auto-gen pickle methods for recursive structs (a bug maybe)
        raise NotImplementedError

    def to_str(self, param_name_fn=None, strip_leading_plus=True):
        if param_name_fn is None:
            param_name_fn = _default_param_name_fn
        result = _tree_node_to_str(&self.data, param_name_fn)
        if strip_leading_plus:
            result = result.lstrip("+")
        return result

    def __str__(self):
        return self.to_str()

    def __repr__(self):
        return f"TreeExpression({self.to_str()})"


cdef _linear_node_to_str(LinearExpressionNode* node, param_name_fn):
    parts = []
    for c, i in zip(node[0].coefficients, node[0].indices):
        if c == 1:
            part = "+" + param_name_fn(i)
        elif c == -1:
            part = "-" + param_name_fn(i)
        else:
            part = f"{c:+g}*" + param_name_fn(i)
        parts.append(part)
    if node[0].bias != 0:
        parts.append(f"{node[0].bias:+g}")
    result = "".join(parts)
    return result


cdef _tree_node_to_str(TreeExpressionNode* node, param_name_fn):
    child_strs = [
        _tree_node_to_str(&node[0].children[i], param_name_fn).lstrip("+")
        for i in range(node[0].children.size())
    ]
    parts = []
    for c, powers, indices in zip(node[0].polynom_coefficients,
                                  node[0].child_powers,
                                  node[0].child_indices):
        multipliers = []
        for power, ch_idx in zip(powers, indices):
            mult_str = child_strs[ch_idx]
            if "*" in mult_str or "+" in mult_str or "-" in mult_str:
                mult_str = "(" + mult_str + ")"
            if power != 1:
                mult_str += f"^{power:g}"
            multipliers.append(mult_str)
        c_str = f"{c:+g}"
        if c_str in ("+1", "-1"):
            c_str = c_str[:1]
        else:
            c_str += "*"
        if multipliers:
            parts.append(c_str + "*".join(multipliers))
        else:
            parts.append(c_str)

    parts.append(_linear_node_to_str(&node[0].linear, param_name_fn))

    return "".join(parts)


def _default_param_name_fn(i):
    i += 1
    parts = []
    while i > 0:
        parts.append(chr(((i-1) % 10) + ord("a")))
        i //= 10
    return "".join(parts)[::-1].capitalize()
