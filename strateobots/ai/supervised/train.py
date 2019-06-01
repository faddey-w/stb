import tensorflow as tf
from tensorflow_estimator import estimator as tf_estimator
from strateobots.ai.lib import data_encoding, data, model_saving
from strateobots.ai import nets
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
        self.controls = {
            ctl: np.argmax(arr, axis=1) if ctl in data.CATEGORICAL_CONTROLS else arr[:, 0]
            for ctl, arr in self.controls.items()
        }
        self.size = self.states.shape[0]
        self.state_dimension = self.states.shape[1]
        # self.control_dimensions = {
        #     ctl: arr.shape[1] for ctl, arr in self.controls.items()
        # }

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
        entropy_weight = params["entropy_weight"]

        model = nets.AnglenavModel(util.get_by_import_path(params["net_ctor"]), params["state_dim"])
        _, predictions = model(state_batch)

        total_loss = None
        train_op = None
        eval_metric_ops = None
        if mode in (tf_estimator.ModeKeys.EVAL, tf_estimator.ModeKeys.TRAIN):
            target_controls = features["controls"]
            losses = {}
            for ctl, ctl_value in target_controls.items():
                pred = predictions[ctl]
                losses[ctl] = tf.reduce_mean(-pred.log_prob(ctl_value))
                # if ctl in data.CATEGORICAL_CONTROLS:
                #     losses[ctl] = tf.Print(losses[ctl], [ctl, ctl_value, pred.probabilities(), losses[ctl]], summarize=20)
                # losses[ctl] -= tf.reduce_mean(pred.entropy() * entropy_weight)
            total_loss = tf.add_n(list(losses.values()))

            if mode is tf_estimator.ModeKeys.EVAL:
                eval_metric_ops = {}
                for ctl, loss in losses.items():
                    eval_metric_ops[f"Loss/{ctl}"] = tf.metrics.mean(loss)
                    if ctl in data.CATEGORICAL_CONTROLS:
                        ctl_feature = data.get_control_feature(ctl)
                        pred_indices = predictions[ctl].choice()
                        pred_probs = predictions[ctl].probabilities()
                        target_indices = target_controls[ctl]
                        for i, label in enumerate(ctl_feature.labels):
                            tgt_match = tf.equal(target_indices, i)
                            pred_match = tf.equal(pred_indices, i)
                            eval_metric_ops[f"ROC-AUC_{ctl}/{label}"] = tf.metrics.auc(
                                tgt_match, pred_probs[..., i]
                            )
                            eval_metric_ops[
                                f"Accuracy_{ctl}/{label}"
                            ] = tf.metrics.accuracy(tgt_match, pred_match)
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
    model_dir = ".data/supervised/models/anglenav3"
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
            "entropy_weight": 0.1,
            "batch_size": 1000,
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
