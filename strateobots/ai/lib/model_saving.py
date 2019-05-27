import inspect
import os
import shutil
import json
import tensorflow as tf
import logging
import hashlib


log = logging.getLogger(__name__)


class ModelManager:
    def __init__(self, model, save_path, src_filepath=None, name_scope=None):
        self.model = model
        self.save_path = _normalize_save_path(save_path)
        if src_filepath is None:
            self._src_filepath = inspect.getsourcefile(model.__class__)
        else:
            self._src_filepath = src_filepath

        assert model.name, "model objects should have a .name"

        self.initializer = tf.variables_initializer(model.var_list)
        if name_scope is None:
            variables = model.var_list
        else:
            variables = {v.op.name[len(name_scope) + 1 :]: v for v in model.var_list}
        self.saver = tf.train.Saver(
            variables,
            pad_step_number=True,
            save_relative_paths=True,
            keep_checkpoint_every_n_hours=0.5,
        )
        self.step_counter = 0

    def save_definition(self, save_path=None):
        if save_path is None:
            save_path = self.save_path
        model = self.model
        os.makedirs(save_path, exist_ok=True)
        with open(_get_construct_params_filepath(save_path), "w") as f:
            construct_params = getattr(model, "construct_params", {})
            params_str = json.dumps(construct_params, indent=4, sort_keys=True)
            f.write(params_str)
        shutil.copy(self._src_filepath, _get_sourcecode_filepath(save_path))
        with open(_get_hash_filepath(save_path), "w") as f:
            f.write(generate_model_hash(self.model, self._src_filepath))

    @classmethod
    def load_existing_model(cls, save_path, name_scope=None):
        with open(_get_construct_params_filepath(save_path)) as f:
            construct_params = json.load(f)
        namespace = {}
        src_filepath = _get_sourcecode_filepath(save_path)
        with open(src_filepath) as f:
            exec(f.read(), namespace)
        if name_scope is not None:
            with tf.variable_scope(name_scope):
                model = namespace["Model"](**construct_params)
        else:
            model = namespace["Model"](**construct_params)
        return cls(model, save_path, src_filepath, name_scope)

    def get_definition_hash(self, save_path=None):
        if save_path is None:
            save_path = self.save_path
        return get_model_definition_hash(save_path)

    def init_vars(self, session=None):
        session.run(self.initializer)

    def save_vars(self, session, inc_step=True, save_path=None):
        if save_path is None:
            save_path = self.save_path
        if not save_path.endswith("/"):
            save_path += "/"
        if inc_step:
            self.step_counter += 1
        self.saver.save(session, save_path, self.step_counter)
        with open(_get_step_filepath(save_path), "w") as f:
            f.write(str(self.step_counter))
        log.info(
            'saved model "%s" at step=%s to %s',
            self.model.name,
            self.step_counter,
            save_path,
        )

    def load_vars(self, session, save_path=None, skip_not_changed=False):
        if save_path is None:
            save_path = self.save_path
        with open(_get_step_filepath(save_path)) as f:
            step_counter = int(f.read().strip())
        if self.step_counter == step_counter and not skip_not_changed:
            log.info(
                "Model is not changed after step=%s, skip loading", self.step_counter
            )
            return
        self.step_counter = step_counter
        ckpt = tf.train.get_checkpoint_state(save_path)
        self.saver.restore(session, ckpt.model_checkpoint_path)
        log.info(
            'loaded model "%s" from %s at step=%s',
            self.model.name,
            ckpt.model_checkpoint_path,
            self.step_counter,
        )


def generate_model_hash(model, model_source_path):
    hasher = hashlib.md5()
    construct_params = getattr(model, "construct_params", {})
    params_str = json.dumps(construct_params, sort_keys=True)
    hasher.update(params_str.encode("utf-8"))
    with open(model_source_path, "rb") as f:
        while True:
            chunk = f.read(4096)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _get_step_filepath(save_path):
    return os.path.join(save_path, "step")


def _get_construct_params_filepath(save_path):
    return os.path.join(save_path, "construct_params.json")


def _get_sourcecode_filepath(save_path):
    return os.path.join(save_path, "definition.py")


def _get_hash_filepath(save_path):
    return os.path.join(save_path, "def-hash")


def _normalize_save_path(save_path):
    if not save_path.endswith(os.path.sep):
        save_path += os.path.sep
    return save_path


def get_model_definition_hash(save_path):
    with open(_get_hash_filepath(save_path)) as f:
        return f.read().strip()
