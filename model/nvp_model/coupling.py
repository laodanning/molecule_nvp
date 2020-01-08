import chainer
import chainer.backends.cuda as cuda
import chainer.functions as F
import chainer.links as L

if __name__ == "__main__":
    import numpy as np
    import os, sys
    sys.path.append(os.getcwd())

from model.nvp_model.mlp import BasicMLP
from model.nvp_model.gat import RelationalGAT
from model.nvp_model.relgcn import RelGCN

class Coupling(chainer.Chain):
    def __init__(self):
        super().__init__()
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def reverse(self):
        raise NotImplementedError


class MolecularCoupling(Coupling):
    def __init__(self, num_nodes, num_relations, num_features, batch_norm=False):
        super(MolecularCoupling, self).__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.num_features = num_features
        self.adj_size = self.num_nodes * self.num_nodes * self.num_relations
        self.x_size = self.num_nodes * self.num_features
        self.apply_bn = batch_norm


class FixedMaskCoupling(MolecularCoupling):
    def __init__(self, num_nodes, num_relations, num_features, mask, batch_norm=False):
        super(FixedMaskCoupling, self).__init__(num_nodes, num_relations, num_features, batch_norm)
        self.mask = mask

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def reverse(self):
        raise NotImplementedError

    def to_gpu(self, device=None):
        device = 0 if device is None else device
        super(FixedMaskCoupling, self).to_gpu(device)
        if isinstance(self.mask, chainer.Variable):
            self.mask.to_gpu(device)
        else:
            self.mask = cuda.to_gpu(self.mask, device)
        if hasattr(self, "cal_mask"):
            self.cal_mask = cuda.to_gpu(self.cal_mask, device)
    
    def to_cpu(self):
        super(FixedMaskCoupling, self).to_cpu()
        if isinstance(self.mask, chainer.Variable):
            self.mask.to_cpu()
        else:
            self.mask = cuda.to_cpu(self.mask)
        if hasattr(self, "cal_mask"):
            self.cal_mask = cuda.to_cpu(self.cal_mask)


class HalfCouplingBlock(chainer.Chain):
    def __init__(self, in_size, out_size, ch_list, batch_norm=False):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.apply_bn = batch_norm

        with self.init_scope():
            self.mlp = BasicMLP(ch_list, in_size=in_size)
            self.linear = L.Linear(ch_list[-1], out_size=2 * self.out_size, initialW=1e-10)
            if self.apply_bn:
                self.batch_norm = L.BatchNormalization(self.in_size)
            self.scale_factor = chainer.Parameter(initializer=0., shape=[1])

    def _s_t_functions(self, x):
        batch_size = x.shape[0]
        x = F.reshape(x, (batch_size, -1))
        assert x.shape[-1] == self.in_size
        if self.apply_bn:
            x = self.batch_norm(x)
        y = F.tanh(self.mlp(x))
        y = self.linear(y) * F.exp(self.scale_factor * 2)
        s = y[:, :self.out_size]
        t = y[:, self.out_size:]
        return s, t

    def __call__(self, x_stable, x_update):
        log_s, t = self._s_t_functions(x_stable)
        x_update = F.reshape(x_update, (x_update.shape[0], -1))
        assert x_update.shape[-1] == self.out_size
        s = F.sigmoid(log_s + 2)
        log_det_jacobian = F.sum(F.log(F.absolute(s)), axis=-1)
        x_update = s * x_update + t
        return x_update, log_det_jacobian
    
    def reverse(self, x_stable, x_update):
        log_s, t = self._s_t_functions(x_stable)
        s = F.sigmoid(log_s + 2)
        x_update = (x_update - t) / s
        return x_update


class RandomShuffleCoupling(Coupling):
    def __init__(self, channel_size, ch_list=None, shuffle=True, batch_norm=False):
        super().__init__()
        self.channel_size = channel_size
        self.apply_bn = batch_norm
        self.apply_shuffle = shuffle
        self.first_half = (channel_size + 1) // 2
        self.second_half = channel_size - self.first_half
        self.ch_list = [256, 512] if ch_list is None else ch_list

        with self.init_scope():
            self.shuffle_idx = self.xp.random.permutation(channel_size).astype(self.xp.int)
            self.add_persistent("{}.shuffle_idx".format(self.name), self.shuffle_idx)
            self.first_part = HalfCouplingBlock(self.first_half, self.second_half, self.ch_list, batch_norm)
            self.second_part = HalfCouplingBlock(self.second_half, self.first_half, self.ch_list, batch_norm)
    
    def __call__(self, z):
        if self.apply_shuffle:
            z = F.permutate(z, self.shuffle_idx, axis=1)
        z1, z2 = F.split_axis(z, [self.first_half], axis=1)
        z2, l1 = self.first_part(z1, z2)
        z1, l2 = self.second_part(z2, z1)
        return F.concat([z1, z2], axis=1), l1 + l2

    def reverse(self, z):
        z1, z2 = F.split_axis(z, [self.first_half], axis=1)
        z1 = self.second_part.reverse(z2, z1)
        z2 = self.first_part.reverse(z1, z2)
        z = F.concat([z1, z2], axis=1)
        return F.permutate(z, self.shuffle_idx, axis=1, inv=True) if self.apply_shuffle else z


class RandomShuffleNodeCoupling(MolecularCoupling):
    def __init__(self, num_nodes, num_relations, num_features, batch_norm=False, shuffle=True, ch_list=None, A_comm=True):
        super().__init__(num_nodes, num_relations, num_features, batch_norm)
        self.apply_shuffle = shuffle
        self.ch_list = ch_list
        self.first_half = (num_features + 1) // 2
        self.second_half = num_features - self.first_half
        self.A_comm = A_comm
        if A_comm:
            self.in_size = [self.first_half * num_nodes + self.adj_size, self.second_half * num_nodes + self.adj_size]
        else:
            self.in_size = [self.first_half * num_nodes, self.second_half * num_nodes]
        self.out_size = [self.second_half * num_nodes, self.first_half * num_nodes]
        
        with self.init_scope():
            self.shuffle_idx = self.xp.random.permutation(num_features).astype(self.xp.int)
            self.add_persistent("{}.shuffle_idx".format(self.name), self.shuffle_idx)
            self.first_part = HalfCouplingBlock(self.in_size[0], self.out_size[0], ch_list, batch_norm)
            self.second_part = HalfCouplingBlock(self.in_size[1], self.out_size[1], ch_list, batch_norm)
    
    def __call__(self, x, A=None):
        if self.apply_shuffle:
            x = F.permutate(x, self.shuffle_idx, axis=2)
        x1, x2 = F.split_axis(x, [self.first_half], axis=2)
        x1_shape = x1.shape
        x2_shape = x2.shape
        x1 = F.reshape(x1, (x1.shape[0], -1))
        x2 = F.reshape(x2, (x2.shape[0], -1))
        if self.A_comm and A is not None:
            A = F.reshape(A, (A.shape[0], -1))
            x1_comp = F.concat([x1, A], axis=1)
            x2, l1 = self.first_part(x1_comp, x2)
            x2_comp = F.concat([x2, A], axis=1)
            x1, l2 = self.second_part(x2_comp, x1)
        else:
            x2, l1 = self.first_part(x1, x2)
            x1, l2 = self.second_part(x2, x1)
        x1 = F.reshape(x1, x1_shape)
        x2 = F.reshape(x2, x2_shape)
        x = F.concat([x1, x2], axis=2)
        return x, l1 + l2

    def reverse(self, x, A=None):
        x1, x2 = F.split_axis(x, [self.first_half], axis=2)
        x1_shape = x1.shape
        x2_shape = x2.shape
        x1 = F.reshape(x1, (x1.shape[0], -1))
        x2 = F.reshape(x2, (x2.shape[0], -1))
        if self.A_comm and A is not None:
            A = F.reshape(A, (A.shape[0], -1))
            x2_comp = F.concat([x2, A], axis=1)
            x1 = self.second_part.reverse(x2_comp, x1)
            x1_comp = F.concat([x1, A], axis=1)
            x2 = self.first_part.reverse(x1_comp, x2)
        else:
            x1 = self.second_part.reverse(x2, x1)
            x2 = self.first_part.reverse(x1, x2)
        x1 = F.reshape(x1, x1_shape)
        x2 = F.reshape(x2, x2_shape)
        x = F.concat([x1, x2], axis=2)
        return F.permutate(x, self.shuffle_idx, axis=2, inv=True) if self.apply_shuffle else x


class RandomShuffleAdjCoupling(MolecularCoupling):
    def __init__(self, num_nodes, num_relations, num_features, batch_norm=False, shuffle=True, ch_list=None, x_comm=True):
        super().__init__(num_nodes, num_relations, num_features, batch_norm=batch_norm)
        self.apply_shuffle = shuffle
        self.ch_list = ch_list
        self.first_half = (num_relations + 1) // 2
        self.second_half = num_relations - self.first_half
        self.x_comm = x_comm
        if x_comm:
            self.in_size = [self.first_half * num_nodes * num_nodes + self.x_size, self.second_half * num_nodes * num_nodes + self.x_size]
        else:
            self.in_size = [self.first_half * num_nodes * num_nodes, self.second_half * num_nodes * num_nodes]
        self.out_size = [self.second_half * num_nodes * num_nodes, self.first_half * num_nodes * num_nodes]

        with self.init_scope():
            self.shuffle_idx = self.xp.random.permutation(num_relations).astype(self.xp.int)
            self.add_persistent("{}.shuffle_idx".format(self.name), self.shuffle_idx)
            self.first_part = HalfCouplingBlock(self.in_size[0], self.out_size[0], ch_list, batch_norm)
            self.second_part = HalfCouplingBlock(self.in_size[1], self.out_size[1], ch_list, batch_norm)
    
    def __call__(self, A, x=None):
        if self.apply_shuffle:
            A = F.permutate(A, self.shuffle_idx, axis=1)
        origin_shape = A.shape
        A1, A2 = F.split_axis(A, [self.first_half], axis=1)
        A1 = F.reshape(A1, (A1.shape[0], -1))
        A2 = F.reshape(A2, (A2.shape[0], -1))
        if self.x_comm and x is not None:
            x = F.reshape(x, (x.shape[0], -1))
            A1_comp = F.concat([A1, x], axis=1)
            A2, l1 = self.first_part(A1_comp, A2)
            A2_comp = F.concat([A2, x], axis=1)
            A1, l2 = self.second_part(A2_comp, A1)
        else:
            A2, l1 = self.first_part(A1, A2)
            A1, l2 = self.second_part(A2, A1)
        A = F.concat([A1, A2], axis=1)
        A = F.reshape(A, origin_shape)
        return A, l1 + l2
    
    def reverse(self, A, x=None):
        origin_shape = A.shape
        A1, A2 = F.split_axis(A, [self.first_half], axis=1)
        A1 = F.reshape(A1, (A1.shape[0], -1))
        A2 = F.reshape(A2, (A2.shape[0], -1))
        if self.x_comm and x is not None:
            x = F.reshape(x, (x.shape[0], -1))
            A2_comp = F.concat([A2, x], axis=1)
            A1 = self.second_part.reverse(A2_comp, A1)
            A1_comp = F.concat([A1, x], axis=1)
            A2 = self.first_part.reverse(A1_comp, A2)
        else:
            A1 = self.second_part.reverse(A2, A1)
            A2 = self.first_part.reverse(A1, A2)
        A = F.concat([A1, A2], axis=1)
        A = F.reshape(A, origin_shape)
        return F.permutate(A, self.shuffle_idx, axis=1, inv=True) if self.apply_shuffle else A


class RandomCommunicateCoupling(MolecularCoupling):
    def __init__(self, num_nodes, num_relations, num_features, batch_norm=False, shuffle=True, A_ch_list=None, x_ch_list=None):
        super().__init__(num_nodes, num_relations, num_features, batch_norm=batch_norm)
        self.A_first = bool(self.xp.random.randint(0, 2))

        with self.init_scope():
            self.A_coupling = RandomShuffleAdjCoupling(num_nodes, num_relations, num_features, batch_norm, shuffle, A_ch_list)
            self.x_coupling = RandomShuffleNodeCoupling(num_nodes, num_relations, num_features, batch_norm, shuffle, x_ch_list)
        
    def __call__(self, x, A):
        if self.A_first:
            A, lA = self.A_coupling(A, x)
            x, lx = self.x_coupling(x, A)
        else:
            x, lx = self.x_coupling(x, A)
            A, lA = self.A_coupling(A, x)
        return x, A, lx, lA

    def reverse(self, x, A):
        if self.A_first:
            x = self.x_coupling.reverse(x, A)
            A = self.A_coupling.reverse(A, x)
        else:
            A = self.A_coupling.reverse(A, x)
            x = self.x_coupling.reverse(x, A)
        return x, A


class RandomCouplingBlock(MolecularCoupling):
    def __init__(self, num_nodes, num_relations, 
                 num_features, num_x_couplings, 
                 num_A_couplings, batch_norm=False, 
                 shuffle=True, x_ch_list=None, A_ch_list=None):
        super().__init__(num_nodes, num_relations, num_features, batch_norm=batch_norm)
        self.num_A_couplings = num_A_couplings
        self.num_x_couplings = num_x_couplings

        with self.init_scope():
            self.comm_coupling = RandomCommunicateCoupling(num_nodes, num_relations, num_features, 
                batch_norm, shuffle, A_ch_list, x_ch_list)
            x_couplings = [RandomShuffleNodeCoupling(num_nodes, num_relations, num_features, 
                batch_norm, shuffle, x_ch_list, A_comm=False) for _ in range(num_x_couplings)]
            self.x_couplings = chainer.ChainList(*x_couplings)
            A_couplings = [RandomShuffleAdjCoupling(num_nodes, num_relations, num_features,
                batch_norm, shuffle, A_ch_list, x_comm=False) for _ in range(num_A_couplings)]
            self.A_couplings = chainer.ChainList(*A_couplings)
    
    def __call__(self, x, A):
        x, A, lx, lA = self.comm_coupling(x, A)
        # lx = chainer.as_variable(
        #     self.xp.zeros([x.shape[0]], dtype=self.xp.float32))
        # lA = chainer.as_variable(
        #     self.xp.zeros([x.shape[0]], dtype=self.xp.float32))
        for i in range(self.num_A_couplings):
            A, l = self.A_couplings[i](A)
            lA += l
        for i in range(self.num_x_couplings):
            x, l = self.x_couplings[i](x)
            lx += l
        return x, A, lx, lA
    
    def reverse(self, x, A):
        for i in reversed(range(self.num_A_couplings)):
            A = self.A_couplings[i].reverse(A)
        for i in reversed(range(self.num_x_couplings)):
            x = self.x_couplings[i].reverse(x)
        x, A = self.comm_coupling.reverse(x, A)
        return x, A


class AffineAdjCoupling(FixedMaskCoupling):
    """
    Mask of adjacency matrix: a boolean ndarray with size (R,)
    """

    def __init__(self, num_nodes, num_relations, num_features,
                 mask, batch_norm=False, coupling_bn=False, ch_list=None):
        super(AffineAdjCoupling, self).__init__(
            num_nodes, num_relations, num_features, mask, batch_norm=batch_norm)
        self.ch_list = ch_list
        self.out_size = num_nodes * num_relations
        self.in_size = self.adj_size - self.out_size

        with self.init_scope():
            self.mlp = BasicMLP(ch_list, in_size=self.in_size, activation=F.relu)
            self.linear = L.Linear(
                ch_list[-1], out_size=2 * self.out_size, initialW=1e-10)
            self.scale_factor = chainer.Parameter(initializer=0., shape=[1])
            self.batch_norm = L.BatchNormalization(self.in_size)

    def __call__(self, adj):
        # adj: (batch_size, num_relations, num_nodes, num_nodes)
        # masked_adj: (batch_size, num_relations - 1, num_nodes, num_nodes)
        masked_adj = adj[:, :, self.mask]
        log_s, t = self._s_t_functions(masked_adj)
        s = F.sigmoid(log_s + 2)
        t = F.broadcast_to(t, adj.shape)
        s = F.broadcast_to(s, adj.shape)

        log_det_jacobian = F.sum(F.log(F.absolute(s)), axis=(1, 2, 3))
        adj = adj * self.mask + adj * \
            (s * ~self.mask) + t * (~self.mask)
        return adj, log_det_jacobian

    def reverse(self, adj):
        masked_adj = adj[:, :, self.mask]
        log_s, t = self._s_t_functions(masked_adj)
        s = F.sigmoid(log_s + 2)
        t = F.broadcast_to(t, adj.shape)
        s = F.broadcast_to(s, adj.shape)
        adj = adj * self.mask + ((adj - t) / s) * (~self.mask)
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
        s = F.reshape(s, [y.shape[0], self.num_relations, self.num_nodes, 1])
        t = F.reshape(t, [y.shape[0], self.num_relations, self.num_nodes, 1])
        return s, t


class AdditiveAdjCoupling(FixedMaskCoupling):
    def __init__(self, num_nodes, num_relations, num_features,
                 mask, batch_norm=False, coupling_bn=False, ch_list=None):
        super(AdditiveAdjCoupling, self).__init__(
            num_nodes, num_relations, num_features, mask, batch_norm=batch_norm)
        self.ch_list = ch_list
        self.out_size = num_nodes * num_relations
        self.in_size = self.adj_size - self.out_size

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
        # adj: (batch_size, num_relations, num_nodes, num_nodes)
        # masked_adj: (batch_size, num_relations - 1, num_nodes, num_nodes)
        masked_adj = adj[:, :, self.mask]
        t = self._s_t_functions(masked_adj)
        t = F.broadcast_to(t, adj.shape)
        adj += t * (~self.mask)

        return adj, chainer.Variable(self.xp.array(0., dtype="float32"))

    def reverse(self, adj):
        masked_adj = adj[:, :, self.mask]
        t = self._s_t_functions(masked_adj)
        t = F.broadcast_to(t, adj.shape)
        adj -= t * (~self.mask)
        return adj, None

    def _s_t_functions(self, adj):
        x = F.reshape(adj, (adj.shape[0], -1))  # flatten
        if self.apply_bn:
            x = self.batch_norm(x)
        y = self.mlp(x)
        y = F.tanh(y)
        y = self.linear(y) * F.exp(self.scale_factor * 2)
        t = F.reshape(y, [y.shape[0], self.num_relations, self.num_nodes, 1])
        return t


class AffineNodeFeatureCoupling(FixedMaskCoupling):
    """
    Mask of adjacency matrix: a boolean ndarray with size (F,)
    """

    def __init__(self, num_nodes, num_relations, num_features, mask,
                 batch_norm=False, coupling_bn=False, ch_list=None, gnn_type="relgcn",
                 gnn_params=None):
        super().__init__(num_nodes, num_relations, num_features, mask, batch_norm=batch_norm)
        self.ch_list = ch_list
        self.out_size = num_features
        self.mask = self.xp.expand_dims(self.mask, axis=-1)
        supported_gnn = ["relgcn", "gat"]
        assert gnn_type in supported_gnn
        with self.init_scope():
            if gnn_type == "relgcn":
                self.gnn = RelGCN(out_channels=ch_list[0], num_edge_type=num_relations, gnn_params=gnn_params)
            elif gnn_type == "gat":
                self.gnn = RelationalGAT(
                    out_dim=ch_list[0], n_edge_types=num_relations, hidden_dim=num_features, gnn_params=gnn_params)
            else:
                raise ValueError("Unsupported GNN type {}, supported types are {}".format(gnn_type, supported_gnn))
            self.linear1 = L.Linear(ch_list[0], out_size=ch_list[1])
            self.linear2 = L.Linear(
                ch_list[1], out_size=2*self.out_size, initialW=1e-10)
            self.scale_factor = chainer.Parameter(initializer=0, shape=[1])
            self.batch_norm = L.BatchNormalization(ch_list[0])

    def __call__(self, x, adj):
        masked_x = self.mask * x
        s, t = self._s_t_functions(masked_x, adj)
        x = masked_x + x * (s * ~self.mask) + t * ~self.mask
        log_det_jacobian = F.sum(F.log(F.absolute(s)), axis=(1, 2))
        return x, log_det_jacobian

    def reverse(self, z, adj):
        masked_z = self.mask * z
        s, t = self._s_t_functions(masked_z, adj)
        x = masked_z + (((z - t)/s) * ~self.mask)
        return x, None

    def _s_t_functions(self, x, adj):
        y = self.gnn(x, adj)
        batch_size = x.shape[0]
        if self.apply_bn:
            y = self.batch_norm(y)
        y = self.linear1(y)
        y = F.tanh(y)
        y = self.linear2(y) * F.exp(self.scale_factor * 2)
        s = y[:, :self.out_size]
        t = y[:, self.out_size:]
        s = F.sigmoid(s + 2)

        t = F.reshape(t, [batch_size, 1, self.out_size])
        t = F.broadcast_to(t, [batch_size, self.num_nodes, self.out_size])
        s = F.reshape(s, [batch_size, 1, self.out_size])
        s = F.broadcast_to(s, [batch_size, self.num_nodes, self.out_size])
        return s, t


class AdditiveNodeFeatureCoupling(FixedMaskCoupling):
    def __init__(self, num_nodes, num_relations, num_features, mask,
                 batch_norm=False, ch_list=None, gnn_type="relgcn",
                 gnn_params=None):
        super().__init__(num_nodes, num_relations, num_features, mask, batch_norm=batch_norm)
        self.ch_list = ch_list
        self.out_size = num_features
        # self.mask = F.expand_dims(self.mask, axis=-1)
        self.mask = self.xp.expand_dims(self.mask, axis=-1)
        supported_gnn = ["relgcn", "gat"]
        assert gnn_type in supported_gnn
        with self.init_scope():
            if gnn_type == "relgcn":
                self.gnn = RelGCN(out_channels=ch_list[0], num_edge_type=num_relations, gnn_params=gnn_params)
            elif gnn_type == "gat":
                self.gnn = RelationalGAT(
                    out_dim=ch_list[0], n_edge_types=num_relations, hidden_dim=num_features, gnn_params=gnn_params)
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
        x += t * (~self.mask)
        return x, chainer.Variable(self.xp.array(0., dtype="float32"))

    def reverse(self, z, adj):
        masked_z = self.mask * z
        batch_size = z.shape[0]
        t = self._s_t_functions(masked_z, adj)
        z -= t * (~self.mask)
        return z, None

    def _s_t_functions(self, x, adj):
        y = self.gnn(x, adj)
        if self.apply_bn:
            y = self.batch_norm(y)
        y = self.linear1(y)
        y = F.tanh(y)
        y = self.linear2(y) * F.exp(self.scale_factor * 2)
        y = F.reshape(y, [x.shape[0], 1, self.out_size])
        y = F.broadcast_to(y, [x.shape[0], self.num_nodes, self.out_size])
        return y


if __name__ == "__main__":
    chainer.config.train = False
    num_nodes = 9
    num_relations = 4
    num_features = 8
    batch_size = 2
    relation_mask = ~np.eye(num_nodes, dtype=np.bool)
    feature_mask = ~np.eye(num_nodes, dtype=np.bool)
    relation_idx = 2
    feature_idx = 3
    ch_list = [64, 128]

    adj_coupling = AffineAdjCoupling(num_nodes, num_relations, num_features, mask=relation_mask[relation_idx], batch_norm=True, ch_list=ch_list)
    adj = np.random.randn(batch_size, num_relations, num_nodes, num_nodes).astype(np.float32)
    adj_z, _ = adj_coupling(adj)
    adj_r, _ = adj_coupling.reverse(adj_z)
    from data.utils import check_reverse
    print("check adj_aff_coupling: {}".format(check_reverse(adj, adj_r)))