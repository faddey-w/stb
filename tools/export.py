import argparse
import os
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from strateobots import util
from strateobots.ai.lib import data_encoding, model_saving, data


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--postprocessing", "-P", choices=["max", "prob"], required=True)
    opts = parser.parse_args(argv)

    controls, model_constructor_path, encoder_name = model_saving.load_model_config(opts.model_dir)
    model_constructor = util.get_by_import_path(model_constructor_path)
    encoder, state_dim = data_encoding.get_encoder(encoder_name)
    checkpoint = tf.train.latest_checkpoint(opts.model_dir)

    state_vec = tf.placeholder(tf.float32, [state_dim], name="Input/state_vector")
    state_vec.set_shape([state_dim])
    state_batch = tf.expand_dims(state_vec, axis=0)
    net = model_constructor(controls)
    logits, predictions = net(state_batch)

    output_nodes = {}
    for ctl, prediction in predictions.items():
        if ctl in data.CATEGORICAL_CONTROLS:
            if opts.postprocessing == "max":
                prediction = tf.argmax(prediction[0])
            elif opts.postprocessing == "prob":
                prediction = tf.random.categorical(logits[ctl], 1)[0, 0]
            else:
                raise NotImplementedError(opts.postprocessing)
            ctl_feature = data.get_control_feature(ctl)
            prediction = tf.constant(ctl_feature.categories)[prediction]
        else:
            prediction = prediction[0]  # just unbatch

        output_nodes[ctl] = prediction
    with tf.name_scope("Output"):
        for ctl, prediction in list(output_nodes.items()):
            output_nodes[ctl] = tf.identity(prediction, name=ctl)

    saver = tf.train.Saver()
    saver_def = saver.as_saver_def()

    os.makedirs(opts.output_dir, exist_ok=True)
    frozen_pb_path = os.path.join(opts.output_dir, "inference.pb")
    freeze_graph.freeze_graph_with_def_protos(
        input_graph_def=tf.get_default_graph().as_graph_def(),
        input_saver_def=saver_def,
        input_checkpoint=checkpoint,
        output_node_names=",".join("Output/" + ctl for ctl in controls),
        restore_op_name="save/restore_all",
        filename_tensor_name="save/Const:0",
        output_graph=frozen_pb_path,
        clear_devices=True,
        initializer_nodes="",
    )
    model_saving.save_model_config(opts.output_dir, controls, model_constructor_path, encoder_name)


if __name__ == '__main__':
    main([
        ".data/supervised/models/anglenav",
        ".data/exported/anglenav",
        "--postprocessing", "prob"
    ])
