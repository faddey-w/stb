import tensorflow as tf


class Linear:
    def __init__(self, name, in_dim, out_dim, shared_weight=None, activation=None):
        self.name = name
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        with tf.variable_scope(self.name):
            self.weight = shared_weight or tf.get_variable("W", [in_dim, out_dim])
            assert self.weight.shape.as_list() == [in_dim, out_dim]
            self.bias = tf.get_variable("B", [1, out_dim])
        if self.weight is shared_weight:
            self.var_list = [self.bias]
        else:
            self.var_list = [self.weight, self.bias]

    def apply(self, x, activation=None):
        if activation is None:
            if self.activation is None:
                raise ValueError("Activation must be specified")
            activation = self.activation
        return self.Apply(x, self, activation)

    class Apply:
        def __init__(self, x, model: "Linear", activation):
            self.name = model.name
            self.model = model
            self.x = x
            with tf.name_scope(self.name):
                self.multiplied = batch_matmul(x, model.weight)
                self.biased = self.multiplied + model.bias
                self.out = activation(self.biased)

    @classmethod
    def chain_factory(cls, input_dim, name_prefix, **kwargs):
        i = 1

        def factory(out_dim, activation=None):
            nonlocal input_dim, i
            name = name_prefix + str(i)
            self = cls(name, input_dim, out_dim, activation=activation, **kwargs)
            i += 1
            input_dim = out_dim
            return self

        return factory


class Residual(Linear):
    def __init__(
        self,
        name,
        in_dim,
        out_dim,
        shared_weight=None,
        activation=None,
        allow_skip_transform=False,
    ):
        super(Residual, self).__init__(name, in_dim, out_dim, shared_weight, activation)
        if in_dim == out_dim and allow_skip_transform:
            self.transform = None
        else:
            with tf.variable_scope(self.name):
                self.transform = tf.get_variable("T", [in_dim, out_dim])
            self.var_list.append(self.transform)

    class Apply:
        def __init__(self, x, model: "Residual", activation):
            self.name = model.name
            self.model = model
            self.x = x
            with tf.name_scope(self.name):
                self.resid = Linear.Apply(x, model, activation)
                self.has_transform = model.transform is not None
                if self.has_transform:
                    self.transformed = batch_matmul(x, model.transform)
                    self.out = self.transformed - self.resid.out
                else:
                    self.out = x - self.resid.out


class ResidualV2:
    def __init__(self, name, in_dim, hidden_dim, out_dim, activation=None):
        self.name = name
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.activation = activation

        self.resid = Linear(name + "/Resid", in_dim, hidden_dim)
        self.join = Linear(name + "/Join", in_dim + hidden_dim, out_dim)

        self.var_list = [*self.resid.var_list, *self.join.var_list]

    def apply(self, x, activation):
        if activation is None:
            if self.activation is None:
                raise ValueError("Activation must be specified")
            activation = self.activation
        return self.Apply(x, self, activation)

    class Apply:
        def __init__(self, x, model: "ResidualV2", activation):
            self.resid = model.resid.apply(x, activation)
            resid_arg = tf.concat([x, self.resid.out], -1)
            self.join = model.join.apply(resid_arg, tf.identity)
            self.out = self.join.out


class ResidualV3:
    def __init__(self, name, n_dim, activation=None):
        self.name = name
        self.n_dim = n_dim
        self.activation = activation

        self.resid = Linear(name + "/Resid", n_dim, n_dim)
        self.mask = tf.get_variable(name + "/M", [n_dim])

        self.var_list = [*self.resid.var_list, self.mask]

    def apply(self, x, activation):
        if activation is None:
            if self.activation is None:
                raise ValueError("Activation must be specified")
            activation = self.activation
        return self.Apply(x, self, activation)

    class Apply:
        def __init__(self, x, model: "ResidualV3", activation):
            self.resid = model.resid.apply(x, activation)
            self.out = x + model.mask * self.resid.out


class LayerChain:
    def __init__(self, factory, *arg_lists, **kwargs):
        self.layers = []
        for args in arg_lists:
            if not isinstance(args, (list, tuple)):
                args = (args,)
            self.layers.append(factory(*args, **kwargs))
        if self.layers and hasattr(self.layers[0], "name"):
            self.name = self.layers[0].name
        self.var_list = sum([l.var_list for l in self.layers], [])

    def apply(self, x, *args, **kwargs):
        nodes = []
        for layer in self.layers:
            node = layer.apply(x, *args, **kwargs)
            x = node.out
            nodes.append(node)
        return nodes


def stack(*layers):
    return LayerChain(lambda layer: layer, *layers)


def shape_to_list(shape):
    if hasattr(shape, "as_list"):
        return shape.as_list()
    else:
        return list(shape)


def batch_matmul(matrix1, matrix2):
    with tf.name_scope("BatchMatMul"):
        return tf.tensordot(matrix1, matrix2, [[-1], [-2]])
