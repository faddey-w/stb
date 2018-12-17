# import tensorflow as tf
import numpy as np
from math import ceil
from strateobots.ai.lib import data


IMAGE_DIM = 100
N_LAYERS = 3
# Layers: hp+velocity+orientation, tower_orientation, bullets


def draw_state(bot, enemy, bot_bullet, enemy_bullet):
    np.empty([IMAGE_DIM, IMAGE_DIM, N_LAYERS], dtype=np.float)


class Model:

    state_dimension = 0
    name = 'VisualModel'

    def __init__(self, name=None):
        if name is not None:
            self.name = name

        with tf.variable_scope(self.name):
            pass
        self.var_list = sum([
        ], [])
        self.init_op = tf.variables_initializer(self.var_list)

    @staticmethod
    @data.generator_encoder
    def data_encoder():
        while True:
            state = yield

    def apply(self, state_vector_array):
        raise NotImplementedError

    class _Apply:

        def __init__(self, state, main, move, rotate, tower_rotate, fire, shield):
            self.state = state
            self.main = main
            self.move = move
            self.rotate = rotate
            self.tower_rotate = tower_rotate
            self.fire = fire
            self.shield = shield
            self.controls = {
                'move': self.move[-1].out,
                'rotate': self.rotate[-1].out,
                'tower_rotate': self.tower_rotate[-1].out,
                'fire': self.fire[-1].out,
                'shield': self.shield[-1].out,
            }


def _draw_gradient_line_py(image, x1, y1, v1, x2, y2, v2, w):
    dx = x2-x1
    dy = y2-y1
    l = (dx*dx + dy*dy) ** 0.5
    cos = dx/l
    sin = dy/l

    w_cos = w*cos
    w_sin = w*sin

    corner_xs, corner_ys = (
        (x1-w_sin, x1+w_sin, x2-w_sin, x2+w_sin),
        (y1+w_cos, y1-w_cos, y2+w_cos, y2-w_cos),
    )
    vo = sorted(range(4), key=corner_xs.__getitem__)
    ascending = corner_ys[vo[1]] < corner_ys[vo[2]]

    # leftmost corner and neighbour sides
    min_x = ceil(corner_xs[vo[0]])
    max_x = int(corner_xs[vo[1]])
    delta_x = max_x - min_x
    y0 = corner_ys[vo[0]]
    if ascending:
        dy_lo = (corner_ys[vo[1]] - y0) / delta_x
        dy_hi = (corner_ys[vo[2]] - y0) / delta_x
    else:
        dy_lo = (corner_ys[vo[2]] - y0) / delta_x
        dy_hi = (corner_ys[vo[1]] - y0) / delta_x
    for i in range(delta_x + 1):
        y_lo = y0 + i * dy_lo
        y_hi = y0 + i * dy_hi
        x = min_x + i
        for y in range(ceil(y_lo), int(y_hi)+1):
            pass
    raise NotImplementedError(...)


def _draw_gradient_line_np(image, x1, y1, v1, x2, y2, v2, w):
    min_x = max(min(x1, x2) - w, 0)
    max_x = min(max(x1, x2) + w, image.shape[0]-1)
    min_y = max(min(y1, y2) - w, 0)
    max_y = min(max(y1, y2) + w, image.shape[1]-1)

    x, y = np.meshgrid(
        np.arange(min_x, max_x+1),
        np.arange(min_y, max_y+1),
    )

    delta_x = x2 - x1
    delta_y = y2 - y1
    line_len = (delta_x**2 + delta_y**2) ** 0.5
    line_dir_x = delta_x / line_len
    line_dir_y = delta_y / line_len

    t = (x - x1) * line_dir_x + (y - y1) * line_dir_y
    r = np.abs((x - x1) * line_dir_y - (y - y1) * line_dir_x)

    v_base = v1 + (v2 - v1) * t / line_len
    v = v_base * (1 - r / w)
    valid_t = (0 < t) & (t <= line_len)
    valid_r = r < w
    mask = valid_r & valid_t

    _draw_values(image, v, mask, min_x, max_x, min_y, max_y)


def _draw_circle_np(image, center_x, center_y, radius, value):
    min_x = max(center_x - radius, 0)
    max_x = min(center_x + radius, image.shape[0]-1)
    min_y = max(center_y - radius, 0)
    max_y = min(center_y + radius, image.shape[1]-1)

    x, y = np.meshgrid(
        np.arange(min_x, max_x + 1),
        np.arange(min_y, max_y + 1),
    )
    x_rel = x - center_x
    y_rel = y - center_y

    r = (x_rel**2 + y_rel**2) ** 0.5
    mask = r <= radius

    _draw_values(image, value, mask, min_x, max_x, min_y, max_y)


def _draw_values(image, value, mask, min_x, max_x, min_y, max_y):
    bg = image[min_x:max_x+1, min_y:max_y+1]
    image[min_x:max_x+1, min_y:max_y+1] = mask * value + (~mask) * bg


if __name__ == '__main__':
    img = np.zeros((100, 100), np.float)
    _draw_circle_np(img, 60, 60, 30, 0.8)
    _draw_gradient_line_np(img, 50, 50, 0.5, 180, 180, 1.0, 5)

    # import code; code.interact(local=globals())

    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    plt.imshow(img, interpolation='nearest')
    plt.show()
