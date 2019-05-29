import tensorflow as tf
from tensorflow_estimator import estimator as tf_estimator
from strateobots.ai.lib import data_encoding, data, model_saving
from strateobots import util
import numpy as np
import os
import logging


class Input:
    def __init__(self, states_dir, controls_dir, game_ids, controls):
        state_arrays = []
        control_arrays = []
        for game_id in game_ids:
            state_arr = np.load(os.path.join(states_dir, game_id)).astype(np.float32)
            ctl_arr = np.load(os.path.join(controls_dir, game_id)).astype(np.float32)
            state_arrays.append(state_arr)
            control_arrays.append(ctl_arr)

        self.states = np.concatenate(state_arrays, axis=0)
        self.controls = np.concatenate(control_arrays, axis=0)
        self.controls = {
            ctl: data_encoding.extract_control(self.controls, ctl) for ctl in controls
        }
        self.size = self.states.shape[0]
        self.state_dimension = self.states.shape[1]
        self.control_dimensions = {
            ctl: arr.shape[1] for ctl, arr in self.controls.items()
        }

    def __call__(self, mode, params):
        if mode not in (tf_estimator.ModeKeys.TRAIN, tf_estimator.ModeKeys.EVAL):
            raise ValueError(mode)
        is_train = mode is tf_estimator.ModeKeys.TRAIN
        ds = tf.data.Dataset.from_tensor_slices(
            {"state": self.states, "controls": self.controls}
        )
        if is_train:
            ds = ds.shuffle(self.size).repeat()
        ds = ds.batch(params["batch_size"], drop_remainder=True)
        return ds


class Model:
    def __call__(self, features, params, mode):
        state_batch = features["state"]  # type: tf.Tensor
        state_batch.set_shape([params["batch_size"], None])

        net = util.get_by_import_path(params["net_ctor"])(params["controls"])
        prediction_logits, predictions = net(state_batch)

        total_loss = None
        train_op = None
        eval_metric_ops = None
        if mode in (tf_estimator.ModeKeys.EVAL, tf_estimator.ModeKeys.TRAIN):
            target_controls = features["controls"]
            losses = {}
            for ctl, ctl_value in target_controls.items():
                if ctl in data.CATEGORICAL_CONTROLS:
                    losses[ctl] = tf.losses.softmax_cross_entropy(
                        ctl_value, prediction_logits[ctl], label_smoothing=0.05
                    )
                else:
                    ctl_value = ctl_value[:, 0]
                    pred = predictions[ctl]
                    if isinstance(pred, tf.distributions.Distribution):
                        mean = pred.mean()
                        stddev = pred.stddev()
                    else:
                        mean = pred
                        stddev = 0
                    losses[ctl] = tf.losses.mean_squared_error(ctl_value, mean) + 0.1 * stddev
            total_loss = tf.add_n(list(map(tf.reduce_mean, losses.values())))

            if mode is tf_estimator.ModeKeys.EVAL:
                eval_metric_ops = {}
                for ctl, loss in losses.items():
                    eval_metric_ops[f"Loss/{ctl}"] = tf.metrics.mean(loss)
                    if ctl in data.CATEGORICAL_CONTROLS:
                        ctl_feature = data.get_control_feature(ctl)
                        for i, label in enumerate(ctl_feature.labels):
                            targets = target_controls[ctl][..., i]
                            preds = predictions[ctl][..., i]
                            eval_metric_ops[f"ROC-AUC_{ctl}/{label}"] = tf.metrics.auc(
                                targets, preds
                            )
                            eval_metric_ops[
                                f"Accuracy_{ctl}/{label}"
                            ] = tf.metrics.accuracy(targets > 0.5, preds > 0.5)
                eval_metric_ops["Loss/Total"] = tf.metrics.mean(total_loss)

            if mode is tf_estimator.ModeKeys.TRAIN:
                for ctl, loss in losses.items():
                    tf.summary.scalar(f"Loss/{ctl}", tf.reduce_mean(loss))
                tf.summary.scalar("Loss/Total", tf.reduce_mean(total_loss))

                optimizer = tf.train.AdamOptimizer()
                train_op = optimizer.minimize(
                    total_loss, global_step=tf.train.get_or_create_global_step()
                )

        return tf_estimator.EstimatorSpec(
            mode=mode,
            # predictions=predictions,
            loss=total_loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
        )


def main():
    states_dir = ".data/supervised/numpy"
    controls_dir = ".data/supervised/controls"
    model_dir = ".data/supervised/models/anglenav2"
    train_steps = 1000

    logging.basicConfig(level=logging.INFO)

    target_controls, model_constructor, _ = model_saving.load_model_config(model_dir)

    game_ids = sorted(os.listdir(states_dir))
    train_input = Input(states_dir, controls_dir, game_ids[:70], target_controls)
    eval_input = Input(states_dir, controls_dir, game_ids[70:], target_controls)
    model = Model()

    estimator = tf_estimator.Estimator(
        model_fn=model,
        params={
            "controls": target_controls,
            "batch_size": 50,
            "state_dim": train_input.state_dimension,
            "net_ctor": model_constructor,
        },
        config=tf_estimator.RunConfig(
            model_dir=model_dir,
            save_checkpoints_steps=100,
            save_summary_steps=20,
            keep_checkpoint_max=5,
        ),
    )
    train_spec = tf_estimator.TrainSpec(input_fn=train_input, max_steps=train_steps)
    eval_spec = tf_estimator.EvalSpec(
        name="0", input_fn=eval_input, steps=None, throttle_secs=1
    )
    tf_estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
    main()
