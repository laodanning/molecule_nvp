import logging as log
from chainer import optimizers
import chainer
import chainer.backends.cuda as cuda
import chainer.functions as F

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
    if str_level == "debug":
        log.basicConfig(level=log.DEBUG)
    elif str_level == "info":
        log.basicConfig(level=log.INFO)
    elif str_level == "warn":
        log.basicConfig(level=log.WARNING)
    elif str_level == "error":
        log.basicConfig(level=log.ERROR)
    elif str_level == "critical":
        log.basicConfig(level=log.CRITICAL)
    else:
        return
    
def get_log_level(str_level: str) -> int:
    return getattr(log, str_level.upper())
