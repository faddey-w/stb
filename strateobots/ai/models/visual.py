import tensorflow as tf
from math import sin, cos
import numpy as np
from math import ceil
from strateobots.ai.lib import data, nn
from strateobots.engine import Constants, BotType


IMAGE_DIM = 32
N_LAYERS = 5
# Layers: (hp+velocity+orientation, tower_orientation) x 2, bullets
_SCALE = IMAGE_DIM / 1000


def draw_state(bot, enemy, bot_bullet, enemy_bullet):
    bot_hp_layer = np.zeros([IMAGE_DIM, IMAGE_DIM], dtype=np.float)
    enemy_hp_layer = np.zeros([IMAGE_DIM, IMAGE_DIM], dtype=np.float)
    bot_gun_layer = np.zeros([IMAGE_DIM, IMAGE_DIM], dtype=np.float)
    enemy_gun_layer = np.zeros([IMAGE_DIM, IMAGE_DIM], dtype=np.float)
    bullets_layer = np.zeros([IMAGE_DIM, IMAGE_DIM], dtype=np.float)
    bullets_buffer = np.empty([IMAGE_DIM, IMAGE_DIM], dtype=np.float)

    _draw_bot(bot_hp_layer, bot_gun_layer, bot)
    _draw_bot(enemy_hp_layer, enemy_gun_layer, enemy)

    for bullet in [bot_bullet, enemy_bullet]:
        if bullet is None:
            continue
        bullets_buffer[...] = 0
        _draw_bullet(bullets_buffer, bullet)
        bullets_layer += bullets_buffer

    return np.stack(
        [bot_hp_layer, enemy_hp_layer, bot_gun_layer, enemy_gun_layer, bullets_layer]
    )


class Model:

    state_dimension = N_LAYERS * IMAGE_DIM * IMAGE_DIM
    name = "VisualModel"

    def __init__(self, name=None):
        if name is not None:
            self.name = name
        self.out_layers = 25

        with tf.variable_scope(self.name):
            self.conv_kernels = [
                tf.get_variable("Conv1", [5, 5, N_LAYERS, 8]),
                tf.get_variable("Conv2", [5, 5, 8, 12]),
                tf.get_variable("Conv3", [3, 3, 12, 16]),
                tf.get_variable("Conv4", [3, 3, 16, 20]),
                tf.get_variable("Conv5", [3, 3, 20, self.out_layers]),
            ]
            self.out_im_size = IMAGE_DIM // (2 ** 5)

            internal_repr_size = 10

            self.ff = nn.LayerChain(
                nn.Linear.chain_factory(
                    self.out_im_size * self.out_im_size * self.out_layers, "FFBlock"
                ),
                (internal_repr_size, tf.sigmoid),
            )
            self.move_block = nn.LayerChain(
                nn.Linear.chain_factory(internal_repr_size, "MoveBlock"),
                # (20, tf.sigmoid),
                # (10, tf.nn.relu),
                (data.ctl_move.dimension, tf.identity),
            )
            self.rotate_block = nn.LayerChain(
                nn.Residual.chain_factory(
                    internal_repr_size, "RotateBlock", allow_skip_transform=True
                ),
                # (20, tf.tanh),
                # (20, tf.nn.relu),
                (data.ctl_rotate.dimension, tf.identity),
            )
            self.tower_rotate_block = nn.LayerChain(
                nn.Linear.chain_factory(internal_repr_size, "TowerBlock"),
                # (20, tf.tanh),
                # (20, tf.nn.relu),
                (data.ctl_tower_rotate.dimension, tf.identity),
                # (data.ctl_tower_rotate.dimension, tf.sin),
            )
            self.action_block = nn.LayerChain(
                nn.Linear.chain_factory(internal_repr_size, "ActionBlock"),
                # (25, tf.sigmoid),
                (data.ctl_action.dimension, tf.identity),
            )
        self.var_list = sum(
            [
                self.conv_kernels,
                self.ff.var_list,
                self.move_block.var_list,
                self.rotate_block.var_list,
                self.tower_rotate_block.var_list,
                self.action_block.var_list,
            ],
            [],
        )
        self.init_op = tf.variables_initializer(self.var_list)

    @staticmethod
    @data.function_encoder
    def data_encoder(bot, enemy, bot_bullet, enemy_bullet):
        img = draw_state(bot, enemy, bot_bullet, enemy_bullet)
        return np.reshape(img, [img.size])

    def apply(self, state_vector_array):
        batch_size = tf.shape(state_vector_array)[0]
        image = tf.reshape(
            state_vector_array, [batch_size, N_LAYERS, IMAGE_DIM, IMAGE_DIM]
        )
        image = tf.transpose(image, (0, 2, 3, 1))

        image_layers = [image]
        for i, kernel in enumerate(self.conv_kernels):
            image = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], "SAME")
            sz = 3 if i < 2 else 2
            image = tf.nn.pool(
                image, [sz, sz], strides=[sz, sz], padding="SAME", pooling_type="MAX"
            )
            image_layers.append(image)
        # import pdb; pdb.set_trace()

        x = tf.reshape(
            image, [batch_size, self.out_layers * self.out_im_size * self.out_im_size]
        )

        ff = self.ff.apply(x)
        return self._Apply(
            image_layers,
            ff,
            self.move_block.apply(ff[-1].out),
            self.rotate_block.apply(ff[-1].out),
            self.tower_rotate_block.apply(ff[-1].out),
            self.action_block.apply(ff[-1].out),
        )

    class _Apply:
        def __init__(self, image_layers, main, move, rotate, tower_rotate, action):
            self.image_layers = image_layers
            self.main = main
            self.move = move
            self.rotate = rotate
            self.tower_rotate = tower_rotate
            self.action = action
            self.controls = {
                "move": self.move[-1].out,
                "rotate": self.rotate[-1].out,
                "tower_rotate": self.tower_rotate[-1].out,
                "action": self.action[-1].out,
            }


def _draw_bot(hp_image, gun_image, bot):
    x = bot["x"] * _SCALE
    y = bot["y"] * _SCALE
    radius = Constants.bot_radius * _SCALE * 1.2
    velocity = (bot["vx"] ** 2 + bot["vy"] ** 2) ** 0.5
    shot_range = BotType.by_code(bot["type"]).shot_range * _SCALE
    angle = bot["orientation"]
    ori_r = 10 + velocity
    _draw_gradient_line_np(
        hp_image, x, y, 1, x + ori_r * cos(angle), y + ori_r * sin(angle), 1, 5
    )
    _draw_circle_np(hp_image, x, y, radius, 0.1 + 0.9 * bot["hp"])

    angle = bot["orientation"] + bot["tower_orientation"]
    _draw_gradient_line_np(
        gun_image,
        x,
        y,
        1,
        x + shot_range * cos(angle),
        y + shot_range * sin(angle),
        1,
        5,
    )


def _draw_bullet(image, bullet):
    x = bullet["x"] * _SCALE
    y = bullet["y"] * _SCALE
    r = bullet["range"] * _SCALE
    angle = bullet["orientation"]
    _draw_gradient_line_np(image, x, y, 1, x + r * cos(angle), y + r * sin(angle), 1, 5)


def _draw_gradient_line_np(image, x1, y1, v1, x2, y2, v2, w):
    # import pdb; pdb.set_trace()
    min_x = int(max(min(x1, x2) - w, 0))
    max_x = ceil(min(max(x1, x2) + w, image.shape[0] - 1))
    min_y = int(max(min(y1, y2) - w, 0))
    max_y = ceil(min(max(y1, y2) + w, image.shape[1] - 1))

    x, y = np.meshgrid(
        np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1), indexing="ij"
    )

    delta_x = x2 - x1
    delta_y = y2 - y1
    line_len = (delta_x ** 2 + delta_y ** 2) ** 0.5
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
    min_x = int(max(center_x - radius, 0))
    max_x = ceil(min(center_x + radius, image.shape[0] - 1))
    min_y = int(max(center_y - radius, 0))
    max_y = ceil(min(center_y + radius, image.shape[1] - 1))

    x, y = np.meshgrid(
        np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1), indexing="ij"
    )
    x_rel = x - center_x
    y_rel = y - center_y

    r = (x_rel ** 2 + y_rel ** 2) ** 0.5
    mask = r <= radius

    _draw_values(image, value, mask, min_x, max_x, min_y, max_y)


def _draw_values(image, value, mask, min_x, max_x, min_y, max_y):
    bg = image[min_x : max_x + 1, min_y : max_y + 1]
    image[min_x : max_x + 1, min_y : max_y + 1] = mask * value + (~mask) * bg


if __name__ == "__main__":
    img = np.zeros((100, 90), np.float)
    # _draw_circle_np(img, 60, 60, 30, 0.8)
    _draw_gradient_line_np(img, 50, 50, 0.5, 180, 180, 1.0, 5)

    # import code; code.interact(local=globals())

    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib import pyplot as plt

    plt.imshow(img, interpolation="nearest")
    plt.show()
