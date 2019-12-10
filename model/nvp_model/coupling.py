import chainer
import chainer.backends.cuda as cuda
import chainer.functions as F
import chainer.links as L

from model.nvp_model.mlp import BasicMLP
from model.nvp_model.gat import RelationalGAT
from model.nvp_model.relgcn import RelGCN


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


class LatentCouplingBlock(chainer.Chain):
    def __init__(self, num_node, num_relation, num_feature, num_coupling, batch_norm=False, shuffle=True, x_ch_list=None, adj_ch_list=None, hint_layer=True):
        super().__init__()
        self.num_node = num_node
        self.num_feature = num_feature
        self.num_relation = num_relation
        self.num_coupling = num_coupling
        self.hint_layer = hint_layer

        with self.init_scope():
            self.x_couplings = chainer.ChainList(*[XLatentCoupling(num_node, num_feature, batch_norm, shuffle, x_ch_list) for _ in range(num_coupling)])
            self.adj_couplings = chainer.ChainList(*[AdjLatentCoupling(num_node, num_relation, batch_norm, shuffle, adj_ch_list) for _ in range(num_coupling)])
            if hint_layer:
                self.hint_coupling = HintCoupling(num_node, num_relation, num_feature, batch_norm, adj_ch_list, x_ch_list)
    
    def __call__(self, x, adj):
        sum_log_det_x = chainer.as_variable(self.xp.zeros(x.shape[0], dtype=self.xp.float32))
        sum_log_det_adj = chainer.as_variable(self.xp.zeros(adj.shape[0], dtype=self.xp.float32))
        for i in range(self.num_coupling):
            x, log_det_x = self.x_couplings[i](x)
            adj, log_det_adj = self.adj_couplings[i](adj)
            sum_log_det_adj += log_det_adj
            sum_log_det_x += log_det_x
        if self.hint_layer:
            x, adj, log_det_x, log_det_adj = self.hint_coupling(x, adj)
            sum_log_det_x += log_det_x
            sum_log_det_adj += log_det_adj
        return x, adj, sum_log_det_x, sum_log_det_adj
    
    def reverse(self, x, adj):
        if self.hint_layer:
            x, adj = self.hint_coupling.reverse(x, adj)
        for i in reversed(range(self.num_coupling)):
            x = self.x_couplings[i].reverse(x)
            adj = self.adj_couplings[i].reverse(adj)
        return x, adj

class AdjLatentCoupling(chainer.Chain):
    def __init__(self, num_node, num_relation, batch_norm=False, shuffle=True, ch_list=None):
        super().__init__()
        self.num_node = num_node
        self.num_relation = num_relation
        self.apply_bn = batch_norm
        self.apply_shuffle = shuffle
        self.ch_list = ch_list
        self.in_half = (num_relation + 1) // 2
        self.out_half = num_relation - self.in_half
        self.in_size = self.in_half * num_node * num_node
        self.out_size = self.out_half * num_node * num_node

        with self.init_scope():
            self.mlp = BasicMLP(ch_list, in_size=self.in_size, activation=F.relu)
            self.shuffle_idx = chainer.Variable(self.xp.random.permutation(num_relation).astype(self.xp.int))
            self.linear = L.Linear(ch_list[-1], out_size=2 * self.out_size, initialW=1e-10)
            self.scale_factor = chainer.Parameter(initializer=0., shape=[1])
            self.batch_norm = L.BatchNormalization(self.in_size)

    def __call__(self, z):
        if self.apply_shuffle:
            z = F.permutate(z, self.shuffle_idx, axis=1)
        z1, z2 = F.split_axis(z, [self.in_half], 1)
        log_s, t = self._s_t_functions(z1)
        s = F.sigmoid(log_s + 2)
        log_det_jacobian = F.sum(F.log(F.absolute(s)), axis=(1, 2, 3))
        z2 = s * z2 + t
        z = F.hstack([z1, z2])
        return z, log_det_jacobian
    
    def reverse(self, z):
        z1, z2 = F.split_axis(z, [self.in_half], 1)
        log_s, t = self._s_t_functions(z1)
        s = F.sigmoid(log_s + 2)
        z2 = (z2 - t) / s
        z = F.hstack([z1, z2])
        return F.permutate(z, self.shuffle_idx, axis=1, inv=True) if self.apply_shuffle else z

    def _s_t_functions(self, z):
        z = F.reshape(z, (z.shape[0], -1))
        x = self.batch_norm(z) if self.apply_bn else z
        y = F.tanh(self.mlp(x))
        y = self.linear(y) * F.exp(self.scale_factor * 2)
        s = y[:, :self.out_size]
        t = y[:, self.out_size:]
        s = F.reshape(s, [s.shape[0], self.out_half, self.num_node, self.num_node])
        t = F.reshape(t, [t.shape[0], self.out_half, self.num_node, self.num_node])
        return s, t


class XLatentCoupling(chainer.Chain):
    def __init__(self, num_node, num_feature, batch_norm=False, shuffle=True, ch_list=None):
        super().__init__()
        self.num_node = num_node
        self.num_feature = num_feature
        self.apply_bn = batch_norm
        self.apply_shuffle = shuffle
        self.ch_list = ch_list
        self.in_half = (num_feature + 1) // 2
        self.out_half = num_feature - self.in_half
        self.in_size = self.in_half * num_node
        self.out_size = self.out_half * num_node

        with self.init_scope():
            self.mlp = BasicMLP(ch_list, in_size=self.in_size, activation=F.relu)
            self.shuffle_idx = chainer.Variable(self.xp.random.permutation(num_feature).astype(self.xp.int))
            self.linear = L.Linear(ch_list[-1], out_size=2 * self.out_size, initialW=1e-10)
            self.scale_factor = chainer.Parameter(initializer=0., shape=[1])
            self.batch_norm = L.BatchNormalization(self.in_size)

    def __call__(self, z):
        if self.apply_shuffle:
            z = F.permutate(z, self.shuffle_idx, axis=2)
        z1, z2 = F.split_axis(z, [self.in_half], -1)
        log_s, t = self._s_t_functions(z1)
        s = F.sigmoid(log_s + 2)
        log_det_jacobian = F.sum(F.log(F.absolute(s)), axis=(1, 2))
        z2 = s * z2 + t
        z = F.concat([z1, z2], axis=-1)
        return z, log_det_jacobian
    
    def reverse(self, z):
        z1, z2 = F.split_axis(z, [self.in_half], -1)
        log_s, t = self._s_t_functions(z1)
        s = F.sigmoid(log_s + 2)
        z2 = (z2 - t) / s
        z = F.concat([z1, z2], axis=-1)
        return F.permutate(z, self.shuffle_idx, axis=2, inv=True) if self.apply_shuffle else z

    def _s_t_functions(self, z):
        z = F.reshape(z, (z.shape[0], -1))
        x = self.batch_norm(z) if self.apply_bn else z
        y = F.tanh(self.mlp(x))
        y = self.linear(y) * F.exp(self.scale_factor * 2)
        s = y[:, :self.out_size]
        t = y[:, self.out_size:]
        s = F.reshape(s, [s.shape[0], self.num_node, self.out_half])
        t = F.reshape(t, [t.shape[0], self.num_node, self.out_half])
        return s, t


class HintCoupling(chainer.Chain):
    def __init__(self, n_nodes, n_relations, n_features, batch_norm=False, adj_ch_list=None, x_ch_list=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_relations = n_relations
        self.n_features = n_features
        self.apply_bn = batch_norm
        self.adj_ch_list = adj_ch_list
        self.x_ch_list = x_ch_list
        self.adj_size = self.n_nodes * self.n_nodes * self.n_relations
        self.x_size = self.n_nodes * self.n_features

        with self.init_scope():
            self.x_hint = XHintCoupling(n_nodes, n_relations, n_features, batch_norm, x_ch_list)
            self.adj_hint = AdjHintCoupling(n_nodes, n_relations, n_features, batch_norm, adj_ch_list)
    
    def __call__(self, x, adj):
        adj, log_det_jac_adj = self.adj_hint(x, adj)
        # x, log_det_jac_x = self.x_hint(x, adj)
        return x, adj, 0, log_det_jac_adj
    
    def reverse(self, x, adj):
        # x = self.x_hint.reverse(x, adj)
        adj = self.adj_hint.reverse(x, adj)
        return x, adj


class XHintCoupling(chainer.Chain):
    """
    A -> z_A, X -> z_X
    s, t = f(z_A)
    z_X' = s * z_X + t
    """
    def __init__(self, n_nodes, n_relations, n_features, batch_norm=False, ch_list=None, additive=False):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_relations = n_relations
        self.n_features = n_features
        self.apply_bn = batch_norm
        self.ch_list = ch_list
        self.adj_size = self.n_nodes * self.n_nodes * self.n_relations
        self.x_size = self.n_nodes * self.n_features
        self.additive = additive

        with self.init_scope():
            self.mlp = BasicMLP(ch_list, in_size=self.adj_size, activation=F.relu)
            if additive:
                self.linear = L.Linear(ch_list[-1], out_size=self.x_size, initialW=1e-10)
            else:
                self.linear = L.Linear(ch_list[-1], out_size=2 * self.x_size, initialW=1e-10)
            self.scale_factor = chainer.Parameter(initializer=0., shape=[1])
            self.batch_norm = L.BatchNormalization(self.adj_size)
        
    def __call__(self, x, adj):
        if self.additive:
            t = self._s_t_functions(adj)
            log_det_jacobian = chainer.Variable(self.xp.array(0., dtype="float32"))
            x += t
        else:
            log_s, t = self._s_t_functions(adj)
            s = F.sigmoid(log_s + 2)
            log_det_jacobian = F.sum(F.log(F.absolute(s)), axis=(1, 2))
            x = x * s + t
        return x, log_det_jacobian
    
    def reverse(self, x, adj):
        if self.additive:
            t = self._s_t_functions(adj)
            x -= t
        else:
            log_s, t = self._s_t_functions(adj)
            s = F.sigmoid(log_s + 2)
            x = (x - t) / s
        return x

    def _s_t_functions(self, adj):
        # x (batch_size, n_relation, n_node, n_node)
        x = F.reshape(adj, (adj.shape[0], -1))
        if self.apply_bn:
            x = self.batch_norm(x)
        y = F.tanh(self.mlp(x))
        y = self.linear(y) * F.exp(self.scale_factor * 2)
        if self.additive:
            y = F.reshape(y, [y.shape[0], self.n_nodes, self.n_features])
            return y
        else:
            s = y[:, :self.x_size]
            t = y[:, self.x_size:]
            s = F.reshape(s, [y.shape[0], self.n_nodes, self.n_features])
            t = F.reshape(t, [y.shape[0], self.n_nodes, self.n_features])
            return s, t

class AdjHintCoupling(chainer.Chain):
    """
    A -> z_A, X -> z_X
    s, t = f(z_X)
    z_A' = s * z_A + t
    """
    def __init__(self, n_nodes, n_relations, n_features, batch_norm=False, ch_list=None, additive=False):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_relations = n_relations
        self.n_features = n_features
        self.apply_bn = batch_norm
        self.ch_list = ch_list
        self.adj_size = self.n_nodes * self.n_nodes * self.n_relations
        self.x_size = self.n_nodes * self.n_features
        self.additive = additive

        with self.init_scope():
            self.mlp = BasicMLP(ch_list, in_size=self.x_size, activation=F.relu)
            if additive:
                self.linear = L.Linear(ch_list[-1], out_size=self.adj_size, initialW=1e-10)
            else:
                self.linear = L.Linear(ch_list[-1], out_size=2 * self.adj_size, initialW=1e-10)
            self.scale_factor = chainer.Parameter(initializer=0., shape=[1])
            self.batch_norm = L.BatchNormalization(self.x_size)

    def __call__(self, x, adj):
        if self.additive:
            t = self._s_t_functions(x)
            log_det_jacobian = chainer.Variable(self.xp.array(0., dtype="float32"))
            adj = adj + t
        else:
            log_s, t = self._s_t_functions(x)
            s = F.sigmoid(log_s + 2)
            log_det_jacobian = F.sum(F.log(F.absolute(s)), axis=(1, 2, 3))
            adj = adj * s + t
        return adj, log_det_jacobian
    
    def reverse(self, x, adj):
        if self.additive:
            t = self._s_t_functions(x)
            adj -= t
        else:
            log_s, t = self._s_t_functions(x)
            s = F.sigmoid(log_s + 2)
            adj = (adj - t) / s
        return adj

    def _s_t_functions(self, x):
        # x (batch_size, n_node, n_feature)
        x = F.reshape(x, (x.shape[0], -1))
        if self.apply_bn:
            x = self.batch_norm(x)
        y = F.tanh(self.mlp(x))
        y = self.linear(y) * F.exp(self.scale_factor * 2)
        if self.additive:
            y = F.reshape(y, [y.shape[0], self.n_relations, self.n_nodes, self.n_nodes])
            return y
        else:
            s = y[:, :self.adj_size]
            t = y[:, self.adj_size:]
            s = F.reshape(s, [y.shape[0], self.n_relations, self.n_nodes, self.n_nodes])
            t = F.reshape(t, [y.shape[0], self.n_relations, self.n_nodes, self.n_nodes])
            return s, t

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
            self.mlp = BasicMLP(ch_list, in_size=self.in_size, activation=F.relu)
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
        t = F.broadcast_to(t, adj.shape)
        s = F.broadcast_to(s, adj.shape)

        log_det_jacobian = F.sum(F.log(F.absolute(s)),
                                 axis=(1, 2, 3))
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
            self.mlp = BasicMLP(ch_list, in_size=self.in_size, activation=F.relu)
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
                 batch_norm=False, ch_list=None, gnn_type="relgcn",
                 gnn_params=None):
        super().__init__(n_nodes, n_relations, n_features, mask, batch_norm=batch_norm)
        self.ch_list = ch_list
        self.out_size = n_nodes
        supported_gnn = ["relgcn", "gat"]
        assert gnn_type in supported_gnn
        with self.init_scope():
            if gnn_type == "relgcn":
                self.gnn = RelGCN(out_channels=ch_list[0], num_edge_type=n_relations, gnn_params=gnn_params)
            elif gnn_type == "gat":
                self.gnn = RelationalGAT(
                    out_dim=ch_list[0], n_edge_types=n_relations, hidden_dim=n_features, gnn_params=gnn_params)
            else:
                raise ValueError("Unsupported GNN type {}, supported types are {}".format(gnn_type, supported_gnn))
            self.linear1 = L.Linear(ch_list[0], out_size=ch_list[1])
            self.linear2 = L.Linear(
                ch_list[1], out_size=2*self.out_size, initialW=1e-10)
            self.scale_factor = chainer.Parameter(initializer=0, shape=[1])
            self.batch_norm = L.BatchNormalization(ch_list[0])

    def __call__(self, x, adj):
        masked_x = self.mask * x  # same shape as x
        batch_size = x.shape[0]

        s, t = self._s_t_functions(masked_x, adj)
        s = F.sigmoid(s + 2)

        t = F.reshape(t, [batch_size, self.out_size, 1])
        t = F.broadcast_to(t, [batch_size, self.out_size, self.n_features])
        s = F.reshape(s, [batch_size, self.out_size, 1])
        s = F.broadcast_to(s, [batch_size, self.out_size, self.n_features])
        log_det_jacobian = F.sum(F.log(F.absolute(s)), axis=(1,2))

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
        y = self.gnn(x, adj)
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
                 batch_norm=False, ch_list=None, gnn_type="relgcn",
                 gnn_params=None):
        super().__init__(n_nodes, n_relations, n_features, mask, batch_norm=batch_norm)
        self.ch_list = ch_list
        self.out_size = n_nodes
        supported_gnn = ["relgcn", "gat"]
        assert gnn_type in supported_gnn
        with self.init_scope():
            if gnn_type == "relgcn":
                self.gnn = RelGCN(out_channels=ch_list[0], num_edge_type=n_relations, gnn_params=gnn_params)
            elif gnn_type == "gat":
                self.gnn = RelationalGAT(
                    out_dim=ch_list[0], n_edge_types=n_relations, hidden_dim=n_features, gnn_params=gnn_params)
            else:
                raise ValueError("Unsupported GNN type {}, supported types are {}".format(gnn_type, supported_gnn))
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
        y = self.gnn(x, adj)
        if self.apply_bn:
            y = self.batch_norm(y)
        y = self.linear1(y)
        y = F.tanh(y)
        y = self.linear2(y) * F.exp(self.scale_factor * 2)

        return y
