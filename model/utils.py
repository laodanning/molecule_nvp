import logging as lg
from chainer import optimizers
import chainer
import chainer.backends.cuda as cuda
import chainer.functions as F

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
    
# class RealNodeMask(chainer.FunctionNode):
#     def __init__(self, virtual_atom_id, output_shape=None, axis=None):
#         super(RealNodeMask, self).__init__()
#         self.virtual_atom_id = virtual_atom_id
#         self.output_shape = output_shape
#         self.axis = axis

#     def forward(self, inputs):
#         # inputs: (batch_size, molecule_size)
#         xp = cuda.get_array_module(inputs)
#         # cond: (batch_size, molecule_size) or (batch_size, molecule_size, feature)
#         cond = xp.not_equal(inputs, self.virtual_atom_id)

def real_node_mask(atom_ids, virtual_atom_id):
    return atom_ids != virtual_atom_id
