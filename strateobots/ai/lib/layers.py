import tensorflow as tf


class Linear:

    class Model:
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

    def __init__(self, name, x, model: 'Linear.Model', activation):
        self.name = name
        self.model = model
        self.x = x
        # batch_shape = tf.shape(x)[:-1]
        # out_shape = tf.concat([batch_shape, [model.out_dim]], 0)
        with tf.name_scope(self.name):
            self.multiplied = batch_matmul(x, model.weight)
            self.biased = self.multiplied + model.bias
            self.out = activation(self.biased)


class Residual:

    class Model(Linear.Model):
        def __init__(self, name, in_dim, out_dim):
            super(Residual.Model, self).__init__(name, in_dim, out_dim)
            with tf.variable_scope(self.name):
                self.transform = tf.get_variable('T', [in_dim, out_dim])
            self.var_list.append(self.transform)

    def __init__(self, name, x, model: 'Residual.Model', activation):
        self.name = name
        self.model = model
        self.x = x
        with tf.name_scope(name):
            self.resid = Linear('Resid', x, model, activation)
            self.transformed = batch_matmul(x, model.transform)
            self.out = self.transformed - self.resid.out


def shape_to_list(shape):
    if hasattr(shape, 'as_list'):
        return shape.as_list()
    else:
        return list(shape)


def batch_matmul(matrix1, matrix2):
    with tf.name_scope('BatchMatMul'):
        return tf.tensordot(matrix1, matrix2, [[-1], [-2]])

