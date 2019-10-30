import chainer
import chainer.backends.cuda as cuda
import chainer.functions as F
import chainer.links as L

from model.nvp_model.mlp import MLP
from model.nvp_model.gat import RelationalGAT


class Coupling(chainer.Chain):
    def __init__(self, n_nodes, n_relations, n_features, mask, batch_norm=False):
        super(Coupling, self).__init__()
        self.mask = mask
        self.n_nodes = n_nodes
        self.n_relations = n_relations
        self.n_features = n_features
        self.adj_size = self.n_nodes * self.n_nodes * self.n_relations
        self.x_size = self.n_nodes * self.n_features
        self.apply_bn = batch_norm

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def reverse(self):
        raise NotImplementedError

    def to_gpu(self, device=None):
        super(Coupling, self).to_gpu(device)
        self.mask = cuda.to_gpu(self.mask, device)
        if hasattr(self, "cal_mask"):
            self.cal_mask = cuda.to_gpu(self.cal_mask, device)
    
    def to_cpu(self):
        super(Coupling, self).to_cpu()
        self.mask = cuda.to_cpu(self.mask)
        if hasattr(self, "cal_mask"):
            self.cal_mask = cuda.to_cpu(self.cal_mask)


class AffineAdjCoupling(Coupling):
    """
    Mask of adjacency matrix: a boolean ndarray with size (R,)
    """

    def __init__(self, n_nodes, n_relations, n_features,
                 mask, batch_norm=False, ch_list=None):
        super(AffineAdjCoupling, self).__init__(
            n_nodes, n_relations, n_features, mask, batch_norm=batch_norm)
        self.ch_list = ch_list
        self.out_size = n_nodes * n_nodes
        self.in_size = self.adj_size - self.out_size
        self.cal_mask = self.mask.reshape(n_relations, 1, 1)

        with self.init_scope():
            self.mlp = MLP(ch_list, in_size=self.in_size, activation=F.relu)
            self.linear = L.Linear(
                ch_list[-1], out_size=2 * self.out_size, initialW=1e-10)
            self.scale_factor = chainer.Parameter(initializer=0., shape=[1])
            self.batch_norm = L.BatchNormalization(self.in_size)

    def __call__(self, adj):
        # adj: (batch_size, n_relations, n_nodes, n_nodes)
        # masked_adj: (batch_size, n_relations - 1, n_nodes, n_nodes)
        masked_adj = adj[:, self.mask]
        log_s, t = self._s_t_functions(masked_adj)
        s = F.sigmoid(log_s + 2)
        log_det_jacobian = F.sum(F.log(F.absolute(s)),
                                 axis=(2, 3)).reshape(adj.shape[0])

        t = F.broadcast_to(t, adj.shape)
        s = F.broadcast_to(s, adj.shape)
        adj = adj * self.cal_mask + adj * \
            (s * ~self.cal_mask) + t * (~self.cal_mask)
        return adj, log_det_jacobian

    def reverse(self, adj):
        masked_adj = adj[:, self.mask]
        log_s, t = self._s_t_functions(masked_adj)
        s = F.sigmoid(log_s + 2)
        t = F.broadcast_to(t, adj.shape)
        s = F.broadcast_to(s, adj.shape)
        adj = adj * self.cal_mask + ((adj - t) / s) * (~self.cal_mask)
        return adj, None

    def _s_t_functions(self, adj):
        x = F.reshape(adj, (adj.shape[0], -1))  # flatten
        if self.apply_bn:
            x = self.batch_norm(x)
        y = self.mlp(x)
        y = F.tanh(y)
        y = self.linear(y) * F.exp(self.scale_factor * 2)
        s = y[:, :self.out_size]
        t = y[:, self.out_size:]
        s = F.reshape(s, [y.shape[0], 1, self.n_nodes, self.n_nodes])
        t = F.reshape(t, [y.shape[0], 1, self.n_nodes, self.n_nodes])
        return s, t


class AdditiveAdjCoupling(Coupling):
    def __init__(self, n_nodes, n_relations, n_features,
                 mask, batch_norm=False, ch_list=None):
        super(AdditiveAdjCoupling, self).__init__(
            n_nodes, n_relations, n_features, mask, batch_norm=batch_norm)
        self.ch_list = ch_list
        self.out_size = n_nodes * n_nodes
        self.in_size = self.adj_size - self.out_size
        self.cal_mask = self.mask.reshape(n_relations, 1, 1)

        with self.init_scope():
            self.mlp = MLP(ch_list, in_size=self.in_size, activation=F.relu)
            self.linear = L.Linear(
                ch_list[-1], out_size=self.out_size, initialW=1e-10)
            self.scale_factor = chainer.Parameter(initializer=0., shape=[1])
            self.batch_norm = L.BatchNormalization(self.in_size)

    def __call__(self, adj):
        """
        Performs one forward step of Adjacency coupling layer.
        Only an additive transformation is applied.
        :param adj: adjacency matrices of molecules
        :return: An adjacency matrix with additive transformation applied (with masking).
            Shape is same as the input adj.
        """
        # adj: (batch_size, n_relations, n_nodes, n_nodes)
        # masked_adj: (batch_size, n_relations - 1, n_nodes, n_nodes)
        masked_adj = adj[:, self.mask]
        t = self._s_t_functions(masked_adj)
        t = F.broadcast_to(t, adj.shape)
        adj += t * (~self.cal_mask)

        return adj, chainer.Variable(self.xp.array(0., dtype="float32"))

    def reverse(self, adj):
        masked_adj = adj[:, self.mask]
        t = self._s_t_functions(masked_adj)
        t = F.broadcast_to(t, adj.shape)
        adj -= t * (~self.cal_mask)
        return adj, None

    def _s_t_functions(self, adj):
        x = F.reshape(adj, (adj.shape[0], -1))  # flatten
        if self.apply_bn:
            x = self.batch_norm(x)
        y = self.mlp(x)
        y = F.tanh(y)
        y = self.linear(y) * F.exp(self.scale_factor * 2)
        t = F.reshape(y, [y.shape[0], 1, self.n_nodes, self.n_nodes])
        return t


class AffineNodeFeatureCoupling(Coupling):
    """
    Mask of adjacency matrix: a boolean ndarray with size (F,)
    """

    def __init__(self, n_nodes, n_relations, n_features, mask,
                 batch_norm=False, ch_list=None, n_attention=4, gat_layers=4):
        super().__init__(n_nodes, n_relations, n_features, mask, batch_norm=batch_norm)
        self.ch_list = ch_list
        self.out_size = n_nodes
        with self.init_scope():
            self.rgat = RelationalGAT(
                out_dim=ch_list[0], n_edge_types=n_relations,
                n_heads=n_attention, hidden_dim=n_features, n_layers=gat_layers)
            self.linear1 = L.linear(ch_list[0], out_size=ch_list[1])
            self.linear2 = L.Linear(
                ch_list[1], out_size=2*self.out_size, initialW=1e-10)
            self.scale_factor = chainer.Parameter(initializer=0, shape=[1])
            self.batch_norm = L.BatchNormalization(ch_list[0])

    def __call__(self, x, adj):
        masked_x = self.mask * x  # same shape as x
        batch_size = x.shape[0]

        s, t = self._s_t_functions(masked_x, adj)
        s = F.sigmoid(s + 2)
        log_det_jacobian = F.sum(F.log(F.absolute(s)), axis=1)

        t = F.reshape(t, [batch_size, self.out_size, 1])
        t = F.broadcast_to(t, [batch_size, self.out_size, self.n_features])
        s = F.reshape(s, [batch_size, self.out_size, 1])
        s = F.broadcast_to(s, [batch_size, self.out_size, self.n_features])

        x = masked_x + x * (s * ~self.mask) + t * ~self.mask
        return x, log_det_jacobian

    def reverse(self, z, adj):
        masked_z = self.mask * z
        batch_size = z.shape[0]
        s, t = self._s_t_functions(masked_z, adj)
        s = F.sigmoid(s + 2)

        t = F.reshape(t, [batch_size, self.out_size, 1])
        t = F.broadcast_to(t, [batch_size, self.out_size, self.n_features])
        s = F.reshape(s, [batch_size, self.out_size, 1])
        s = F.broadcast_to(s, [batch_size, self.out_size, self.n_features])
        out = masked_z + ((z - t) / s) * (~self.mask)

        return out, None

    def _s_t_functions(self, x, adj):
        y = self.rgat(x, adj)
        if self.apply_bn:
            y = self.batch_norm(y)
        y = self.linear1(y)
        y = F.tanh(y)
        y = self.linear2(y) * F.exp(self.scale_factor * 2)
        s = y[:, :self.out_size]
        t = y[:, self.out_size:]

        return s, t


class AdditiveNodeFeatureCoupling(Coupling):
    def __init__(self, n_nodes, n_relations, n_features, mask,
                 batch_norm=False, ch_list=None, n_attention=4, gat_layers=4):
        super().__init__(n_nodes, n_relations, n_features, mask, batch_norm=batch_norm)
        self.ch_list = ch_list
        self.out_size = n_nodes
        with self.init_scope():
            self.rgat = RelationalGAT(
                out_dim=ch_list[0], n_edge_types=n_relations,
                n_heads=n_attention, hidden_dim=n_features, n_layers=gat_layers)
            self.linear1 = L.Linear(ch_list[0], out_size=ch_list[1])
            self.linear2 = L.Linear(
                ch_list[1], out_size=self.out_size, initialW=1e-10)
            self.scale_factor = chainer.Parameter(initializer=0, shape=[1])
            self.batch_norm = L.BatchNormalization(ch_list[0])

    def __call__(self, x, adj):
        masked_x = self.mask * x
        batch_size = x.shape[0]
        t = self._s_t_functions(masked_x, adj)
        t = F.reshape(t, [batch_size, self.out_size, 1])
        t = F.broadcast_to(t, [batch_size, self.out_size, self.n_features])
        x += t * (~self.mask)
        return x, chainer.Variable(self.xp.array(0., dtype="float32"))

    def reverse(self, z, adj):
        masked_z = self.mask * z
        batch_size = z.shape[0]
        t = self._s_t_functions(masked_z, adj)
        t = F.reshape(t, [batch_size, self.out_size, 1])
        t = F.broadcast_to(t, [batch_size, self.out_size, self.n_features])
        z -= t * (~self.mask)
        return z, None

    def _s_t_functions(self, x, adj):
        y = self.rgat(x, adj)
        if self.apply_bn:
            y = self.batch_norm(y)
        y = self.linear1(y)
        y = F.tanh(y)
        y = self.linear2(y) * F.exp(self.scale_factor * 2)

        return y
