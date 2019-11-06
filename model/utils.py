import logging as log
from chainer import optimizers
import chainer
import chainer.backends.cuda as cuda
import chainer.functions as F
from model.nvp_model.nvp_model import AttentionNvpModel

def get_and_log(config, key, default_value=None, required=False):
    value = config.get(key, default_value)
    if required and value is None:
        raise ValueError("{} value must be given.".format(key))
    log.info("{}:\t{}".format(key, value))
    return value

def get_optimizer(opt_type: str):
    if opt_type == "adam":
        return optimizers.Adam
    elif opt_type == "momentum":
        return optimizers.MomentumSGD
    elif opt_type == "sgd":
        return optimizers.SGD
    elif opt_type == "rmsprop":
        return optimizers.RMSprop
    else:
        log.error("Unsupported optmizer {}!".format(opt_type))
        return None

def real_node_mask(atom_ids, virtual_atom_id):
    return atom_ids != virtual_atom_id

def set_log_level(str_level: str) -> None:
    log.basicConfig(level=get_log_level(str_level))
    
def get_log_level(str_level: str) -> int:
    return getattr(log, str_level.upper())

def load_model_from(path: str, model_params) -> chainer.Chain:
    log.info("loading model from '{}'".format(path))
    log.debug("Hyperparams: \n{}\n".format(model_params))
    model = AttentionNvpModel(model_params)
    if path.endswith(".npz"):
        chainer.serializers.load_npz(path, model)
    else:
        chainer.serializers.load_npz(
            path, model, path="updater/optimizer:main/", strict=False)
    return model
