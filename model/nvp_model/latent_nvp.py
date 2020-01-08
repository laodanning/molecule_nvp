import chainer
import chainer.links as L
import chainer.functions as F
import math

if __name__ == "__main__":
    import numpy as np
    import os, sys
    sys.path.append(os.getcwd())

from model.hyperparameter import Hyperparameter
from model.nvp_model.coupling import RandomShuffleCoupling


class LatentNVP(chainer.Chain):
    def __init__(self, hyperparams):
        super().__init__()

        # read parameters
        self.num_coupling_layers = hyperparams.num_coupling_layers
        self.ch_list = hyperparams.ch_list
        self._learn_var = hyperparams.learn_var
        self.channel_size = hyperparams.channel_size

        with self.init_scope():
            couplings = [RandomShuffleCoupling(self.channel_size, self.ch_list) for _ in range(self.num_coupling_layers)]
            self.couplings = chainer.ChainList(*couplings)
            self.dist_distance_scale = chainer.Parameter(initializer=1., shape=[1])
            if self._learn_var:
                self.pos_var = chainer.Parameter(initializer=0., shape=[1])
                self.neg_var = chainer.Parameter(initializer=0., shape=[1])
            else:
                self.pos_var = chainer.Variable(initializer=0., shape=[1])
                self.neg_var = chainer.Variable(initializer=0., shape=[1])       

    def __call__(self, z):
        z = chainer.as_variable(z)
        batch_size = z.shape[0]
        sum_log_det_jacobian = chainer.as_variable(
            self.xp.zeros([batch_size], dtype=self.xp.float32))

        for i in range(self.num_coupling_layers):
            z, l = self.couplings[i](z)
            sum_log_det_jacobian += l

        return z, sum_log_det_jacobian

    def reverse(self, z):
        with chainer.no_backprop_mode():
            for i in reversed(range(self.num_coupling_layers)):
                z = self.couplings[i].reverse(z)
        print(z)
        return z

    def nll(self, z, t, sum_log_det_jacobian):
        # t: 0 for neg and 1 for pos
        sum_log_det_jacobian -= self.channel_size
        broadcasted_t = F.broadcast_to(F.reshape(t, [t.shape[0], 1]), [t.shape[0], self.channel_size])
        var = F.where(
                F.cast(broadcasted_t, self.xp.bool), 
                self.xp.ones(self.channel_size) * self.pos_var, 
                self.xp.ones(self.channel_size) * self.neg_var)
        broadcasted_t = 2  * (F.cast(broadcasted_t, self.xp.float32) - 0.5) * self.dist_distance_scale # -1 for neg and 1 for pos
        raw_data_nll = F.average(F.sum(F.gaussian_nll(z, broadcasted_t, var, reduce="no"), axis=1) - sum_log_det_jacobian)
        raw_data_nll /= self.channel_size
        
        return raw_data_nll
    
    @property
    def latent_size(self):
        return self.channel_size
    
    @property
    def learn_var(self):
        return self._learn_var

    @property
    def ln_var(self):
        return self.xp.ones(self.channel_size) * self.pos_var

    @property
    def mean(self):
        return self.xp.ones(self.channel_size, dtype=self.xp.float32) * self.dist_distance_scale.array


if __name__ == "__main__":
    params_dict = {
        "channel_size": 10,
        "ch_list": [16, 16],
        "learn_var": True,
        "num_coupling_layers": 3
    }
    hyper = Hyperparameter()
    hyper._parse_dict(params_dict)
    chainer.config.train = False


    batch_size = 8
    raw_std = 0.1
    LNVP = LatentNVP(hyper)
    x = np.random.randn(batch_size, params_dict["channel_size"]).astype(np.float32) * raw_std
    z, l = LNVP(x)
    x_r = LNVP.reverse(z)
    diff = x - x_r

    print(z)
    print(x)

    # # check reverse == raw
    # print("reverse check: {}".format(np.prod(1 - diff).array.astype(bool)))
    
    # # check nll function
    # t = np.random.randint(0, 2, size=[batch_size])
    # nll = LNVP.nll(z, t, l)
    # print(nll)
