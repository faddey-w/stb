import inspect
import os
import shutil
import json
import tensorflow as tf
import logging


log = logging.getLogger(__name__)


class ModelManager:

    def __init__(self, model, save_path, _need_initial_save=True):
        self.model = model
        self.save_path = _normalize_save_path(save_path)

        assert model.name, "model objects should have a .name"

        if _need_initial_save:
            os.makedirs(self.save_path, exist_ok=True)
            with open(_get_construct_params_filepath(save_path), 'w') as f:
                f.write(json.dumps(getattr(model, 'construct_params', {})))
            src_filepath = inspect.getsourcefile(model.__class__)
            shutil.copy(src_filepath, _get_sourcecode_filepath(save_path))

        self.initializer = tf.variables_initializer(model.var_list)
        self.saver = tf.train.Saver(
            model.var_list,
            pad_step_number=True,
            save_relative_paths=True,
        )
        self.step_counter = 0

    @classmethod
    def load_existing_model(cls, save_path):
        with open(_get_construct_params_filepath(save_path)) as f:
            construct_params = json.load(f)
        namespace = {}
        with open(_get_sourcecode_filepath(save_path)) as f:
            exec(f.read(), namespace)
        model = namespace['Model'](**construct_params)
        return cls(model, save_path, _need_initial_save=False)

    def init_vars(self, session=None):
        session.run(self.initializer)

    def save_vars(self, session):
        self.step_counter += 1
        self.saver.save(session, self.save_path, self.step_counter)
        with open(_get_step_filepath(self.save_path), 'w') as f:
            f.write(str(self.step_counter))
        log.info('saved model "%s" at step=%s to %s',
                 self.model.name, self.step_counter, self.save_path)

    def load_vars(self, session):
        with open(_get_step_filepath(self.save_path)) as f:
            step_counter = int(f.read().strip())
        if self.step_counter == step_counter:
            log.info("Model is not changed after step=%s, skip loading", self.step_counter)
            return
        self.step_counter = step_counter
        ckpt = tf.train.get_checkpoint_state(self.save_path)
        log.info('loading model "%s" from %s at step=%s',
                 self.model.name, ckpt.model_checkpoint_path, self.step_counter)
        self.saver.restore(session, ckpt.model_checkpoint_path)


def _get_step_filepath(save_path):
    return os.path.join(save_path, 'step')


def _get_construct_params_filepath(save_path):
    return os.path.join(save_path, 'construct_params.json')


def _get_sourcecode_filepath(save_path):
    return os.path.join(save_path, 'definition.py')


def _normalize_save_path(save_path):
    if not save_path.endswith(os.path.sep):
        save_path += os.path.sep
    return save_path
