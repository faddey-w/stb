# distutils: language=c++
from libcpp.vector cimport vector
from libc.math cimport pow


cdef struct LinearExpressionNode:
    vector[int] indices
    vector[float] coefficients
    float bias


cdef struct TreeExpressionNode:
    # Let N - amount of env parameters
    #     M - amount of children expressions
    #     P - length of polynom built from children expressions
    #     K_p - amount of factors in p-th polynom member
    vector[TreeExpressionNode] children  # M
    vector[vector[int]] child_indices  # P x K_p
    vector[vector[float]] child_powers  # P x K_p
    vector[float] polynom_coefficients  # P
    LinearExpressionNode linear


cdef float _evaluate_linear_node(LinearExpressionNode* node, vector[float]* params):
    cdef float result = 0.0
    for n in range(node[0].coefficients.size()):
        result += node[0].coefficients[n] * params[0][node[0].indices[n]]
    result += node[0].bias
    return result


cdef float _evaluate_tree_node(TreeExpressionNode* node, vector[float]* params):
    cdef vector[float] values
    cdef float result
    cdef float r

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


cdef class LinearExpression:
    cdef LinearExpressionNode data

    def __init__(self):
        self.data.indices.push_back(0)
        self.data.indices.push_back(1)
        self.data.indices.push_back(2)
        self.data.coefficients.push_back(10.0)
        self.data.coefficients.push_back(2.0)
        self.data.coefficients.push_back(0.5)
        self.data.bias = +100.0

    cpdef float evaluate(self, vector[float] params):
        return _evaluate_linear_node(&self.data, &params)


cdef class TreeExpression:

    cdef TreeExpressionNode data

    cpdef float evaluate(self, vector[float] params):
        return _evaluate_tree_node(&self.data, &params)

    def __reduce__(self):  # Cython can't auto-gen pickle methods for recursive structs
        raise NotImplementedError


