import tensorflow as tf
from tensorflow.python.framework import function


EPS = 0.001
MAX_GRAD = 5


@function.Defun(tf.float32, tf.float32, tf.float32)
def l2norm_grad(x, y, dr):
    r = tf.sqrt(x*x + y*y)
    mask = tf.greater(r, EPS)
    dx = dr * tf.where(mask, x / r, tf.ones_like(r))
    dy = dr * tf.where(mask, y / r, tf.ones_like(r))
    return dx, dy


@function.Defun(tf.float32, tf.float32, grad_func=l2norm_grad)
def l2norm(x, y):
    return tf.sqrt(x*x + y*y)


@function.Defun(tf.float32, tf.float32, tf.float32)
def atan2_grad(y, x, dtheta):
    r2 = y*y + x*x
    r2 = tf.maximum(r2, EPS)
    mask = tf.greater(r2, EPS)
    dy = tf.where(mask, dtheta * x / r2, MAX_GRAD * tf.sign(dtheta * x))
    dx = tf.where(mask, - dtheta * y / r2, MAX_GRAD * tf.sign(- dtheta * y))
    return dy, dx


@function.Defun(tf.float32, tf.float32, grad_func=atan2_grad)
def atan2(y, x):
    return tf.atan2(y, x)


def _test():
    arg_y = tf.constant([0.1, 0.05, 0.01], dtype=tf.float32)
    arg_x = tf.constant([0.1, 0.05, 0.01], dtype=tf.float32)
    grad_tf = tf.gradients(tf.atan2(arg_y, arg_x), [arg_x, arg_y])
    grad_st = tf.gradients(atan2(arg_y, arg_x), [arg_x, arg_y])
    sess = tf.Session()
    print(sess.run(grad_tf))
    print(sess.run(grad_st))


if __name__ == '__main__':
    _test()
