import logging as lg
from chainer import optimizers

def get_and_log(config, key, default_value=None, required=False):
    value = config.get(key, default_value)
    if required and value is None:
        raise ValueError("{} value must be given.".format(key))
    lg.info("{}:\t{}".format(key, value))
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
        lg.error("Unsupported optmizer {}!".format(opt_type))
        return None
    