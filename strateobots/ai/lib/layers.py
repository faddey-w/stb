import tensorflow as tf


class Linear:

    def __init__(self, name, in_dim, out_dim, shared_weight=None):
        self.name = name
        self.in_dim = in_dim
        self.out_dim = out_dim
        with tf.variable_scope(self.name):
            self.weight = shared_weight or tf.get_variable('W', [in_dim, out_dim])
            assert self.weight.shape.as_list() == [in_dim, out_dim]
            self.bias = tf.get_variable('B', [1, out_dim])
        if self.weight is shared_weight:
            self.var_list = [self.bias]
        else:
            self.var_list = [self.weight, self.bias]

    def apply(self, x, activation):
        return self.Apply(x, self, activation)

    class Apply:
        def __init__(self, x, model: 'Linear', activation):
            self.name = model.name
            self.model = model
            self.x = x
            with tf.name_scope(self.name):
                self.multiplied = batch_matmul(x, model.weight)
                self.biased = self.multiplied + model.bias
                self.out = activation(self.biased)


class Residual(Linear):

    def __init__(self, name, in_dim, out_dim):
        super(Residual, self).__init__(name, in_dim, out_dim)
        with tf.variable_scope(self.name):
            self.transform = tf.get_variable('T', [in_dim, out_dim])
        self.var_list.append(self.transform)

    class Apply:
        def __init__(self, x, model: 'Residual', activation):
            self.name = model.name
            self.model = model
            self.x = x
            with tf.name_scope(self.name):
                self.resid = Linear.Apply(x, model, activation)
                self.transformed = batch_matmul(x, model.transform)
                self.out = self.transformed - self.resid.out


class ResidualV2:

    def __init__(self, name, in_dim, hidden_dim, out_dim):
        self.name = name
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.resid = Linear(name + '/Resid', in_dim, hidden_dim)
        self.join = Linear(name + '/Join', in_dim + hidden_dim, out_dim)

        self.var_list = [*self.resid.var_list, *self.join.var_list]

    def apply(self, x, activation):
        return self.Apply(x, self, activation)

    class Apply:
        def __init__(self, x, model: 'ResidualV2', activation):
            self.resid = model.resid.apply(x, activation)
            resid_arg = tf.concat([x, self.resid.out], -1)
            self.join = model.join.apply(resid_arg, tf.identity)
            self.out = self.join.out


class ResidualV3:

    def __init__(self, name, n_dim):
        self.name = name
        self.n_dim = n_dim

        self.resid = Linear(name + '/Resid', n_dim, n_dim)
        self.mask = tf.get_variable(name + '/M', [n_dim])

        self.var_list = [*self.resid.var_list, self.mask]

    def apply(self, x, activation):
        return self.Apply(x, self, activation)

    class Apply:
        def __init__(self, x, model: 'ResidualV3', activation):
            self.resid = model.resid.apply(x, activation)
            self.out = x + model.mask * self.resid.out


# class LayerChain:
#
#     def __init__(self, factory, *args, **kwargs):
#         self.layers = []
#         while True:
#             layer, args, kwargs = factory(*args, **kwargs)
#             if layer is not None:
#                 self.layers.append(layer)
#             else:
#                 break
#         if self.layers and hasattr(self.layers[0], 'name'):
#             self.name = self.layers[0].name
#
#     def apply(self, x, *args, **kwargs):
#         nodes = []
#         for layer in self.layers:
#             node = layer.apply(x, *args, **kwargs)
#             x = node.out
#             nodes.append(node)
#         return self._Apply(nodes)
#
#     class _Apply:
#         def __init__(self, nodes):
#             self.nodes = nodes
#
#
# def simple_chain(layer_cls, name_prefix, param0, other_params):
#


def shape_to_list(shape):
    if hasattr(shape, 'as_list'):
        return shape.as_list()
    else:
        return list(shape)


def batch_matmul(matrix1, matrix2):
    with tf.name_scope('BatchMatMul'):
        return tf.tensordot(matrix1, matrix2, [[-1], [-2]])

