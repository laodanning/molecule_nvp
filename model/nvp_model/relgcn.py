import chainer
import chainer.functions as F
from chainer import cuda
from chainer_chemistry import MAX_ATOMIC_NUM
from chainer_chemistry.links import EmbedAtomID
from chainer_chemistry.links import GraphLinear


class RelGCNUpdate(chainer.Chain):

    def __init__(self, in_channels, out_channels, num_edge_type=4):
        """
        """
        super(RelGCNUpdate, self).__init__()
        with self.init_scope():
            self.graph_linear_self = GraphLinear(in_channels, out_channels)
            self.graph_linear_edge = GraphLinear(in_channels, out_channels * num_edge_type)
        self.num_edge_type = num_edge_type
        self.in_ch = in_channels
        self.out_ch = out_channels

    def __call__(self, h, adj):
        """

        Args:
            h:
            adj:

        Returns:

        """

        mb, node, ch = h.shape

        # --- self connection, apply linear function ---
        hs = self.graph_linear_self(h)
        # --- relational feature, from neighbor connection ---
        # Expected number of neighbors of a vertex
        # Since you have to divide by it, if its 0, you need to arbitrarily set it to 1
        m = self.graph_linear_edge(h)
        m = F.reshape(m, (mb, node, self.out_ch, self.num_edge_type))
        m = F.transpose(m, (0, 3, 1, 2))
        # m: (batchsize, edge_type, node, ch)
        # hr: (batchsize, edge_type, node, ch)
        hr = F.matmul(adj, m)
        # hr: (batchsize, node, ch)
        hr = F.sum(hr, axis=1)
        return hs + hr


class RelGCNReadout(chainer.Chain):

    """RelGCN submodule for updates"""

    def __init__(self, in_channels, out_channels, nobias=True):
        super(RelGCNReadout, self).__init__()
        with self.init_scope():
            self.sig_linear = GraphLinear(
                in_channels, out_channels, nobias=nobias)
            self.tanh_linear = GraphLinear(
                in_channels, out_channels, nobias=nobias)

    def __call__(self, h, x=None):
        """Relational GCN

        (implicit: N = number of edges, R is number of types of relations)
        Args:
            h (chainer.Variable): (batchsize, num_nodes, ch)
                N x F : Matrix of edges, each row is a molecule and each column is a feature.
                F_l is the number of features at layer l
                F_0, the input layer, feature is type of molecule. Softmaxed

            x (chainer.Variable): (batchsize, num_nodes, ch)

        Returns:
            h_n (chainer.Variable): (batchsize, ch)
                F_n : Graph level representation

        Notes: I think they just incorporate "no edge" as one of the categories of relations, i've made it a separate
            tensor just to simplify some implementation, might change later
        """
        if x is None:
            in_feat = h
        else:
            in_feat = F.concat([h, x], axis=2)
        sig_feat = F.sigmoid(self.sig_linear(in_feat))
        tanh_feat = F.tanh(self.tanh_linear(in_feat))

        return F.tanh(F.sum(sig_feat * tanh_feat, axis=1))


class RelGCN(chainer.Chain):

    def __init__(self, out_channels=64, num_edge_type=4,
                 activation=F.tanh, gnn_params=None):

        super(RelGCN, self).__init__()
        if gnn_params is None:
            gnn_params = {}
        self.ch_list = gnn_params.get("ch_list", [16, 128, 64])
        self.scale_adj = gnn_params.get("scale_adj", False)
        ch_list = self.ch_list

        with self.init_scope():
            self.embed = GraphLinear(None, ch_list[0])
            self.rgcn_convs = chainer.ChainList(*[
                RelGCNUpdate(ch_list[i], ch_list[i+1], num_edge_type) for i in range(len(ch_list)-1)])
            self.rgcn_readout = RelGCNReadout(ch_list[-1], out_channels)
        self.activation = activation

    def rescale_adj(self, adj):
        num_neighbors = F.sum(adj, axis=(1, 2))
        base = self.xp.ones(num_neighbors.shape, dtype=self.xp.float32)
        cond = num_neighbors.data != 0
        num_neighbors_inv = 1 / F.where(cond, num_neighbors, base)
        return adj * F.broadcast_to(num_neighbors_inv[:, None, None, :], adj.shape)    

    def __call__(self, x, adj):
        """

        Args:
            x: (batchsize, num_nodes, in_channels)
            adj: (batchsize, num_edge_type, num_nodes, num_nodes)

        Returns: (batchsize, out_channels)

        """
        h = self.embed(x)  # (minibatch, max_num_atoms)
        if self.scale_adj:
            adj = self.rescale_adj(adj)
        for rgcn_conv in self.rgcn_convs:
            h = self.activation(rgcn_conv(h, adj))
        h = self.rgcn_readout(h)
        return h
