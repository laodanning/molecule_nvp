import chainer
from chainer import cuda
import chainer.functions as F
from chainer_chemistry import MAX_ATOMIC_NUM
from chainer_chemistry.links import EmbedAtomID
from chainer_chemistry.links import GraphLinear

def rescale_adj(adj):
    """Rescale the adjacency matrix
    Args:
        adj (xp.ndarray): adjacency matrix with shape (batch_size, num_edge_type, num_node, num_node)
    Returns:
        A scaled adjacency matrix
    """
    xp = cuda.get_array_module(adj)
    num_neighbors = F.sum(adj, axis=(1, 2))
    batch_size, num_edge_type, num_node, _ = adj.shape
    base = xp.ones(num_neighbors.shape, dtype=xp.float32)
    cond = num_neighbors.array != 0
    num_neighbors_inv = F.reshape(1 / F.where(cond, num_neighbors, base), (batch_size, 1, 1, num_node))
    return adj * F.broadcast_to(num_neighbors_inv, adj.shape)

class AtomEmbed(chainer.Chain):
    """Implementation for atom embedding

    Args:
        word_size (int): output size, size of 'atom word'
        num_atom_type (int): num of id, default is MAX_ATOMIC_NUM
        id_trans_fn (chainer.FunctionNode): function that transfer raw data id to id used in embedding
    """

    def __init__(self, word_size, num_atom_type=MAX_ATOMIC_NUM, id_trans_fn=None):
        super(AtomEmbed, self).__init__()
        with self.init_scope():
            self.embed = EmbedAtomID(out_size=word_size, in_size=num_atom_type)
        self.word_size = word_size
        self.num_atom_type = num_atom_type
        self.id_trans_fn = id_trans_fn
        
    def __call__(self, x):
        if self.id_trans_fn is None:
            words = self.embed(x)
        else:
            words = self.embed(self.id_trans_fn(x))
        return words


class AtomEmbedRGCNUpdate(chainer.Chain):

    def __init__(self, in_channel, out_channel, num_edge_type=4):
        super(AtomEmbedRGCNUpdate, self).__init__()
        with self.init_scope():
            self.graph_linear_edge = GraphLinear(in_channel, out_channel * num_edge_type)
            self.graph_linear_self = GraphLinear(in_channel, out_channel)
        self.num_edge_type = num_edge_type
        self.in_channel = in_channel
        self.out_channel = out_channel

    def __call__(self, h, adj):
        batch_size, num_node, num_channel = h.shape

        # --- self connection ---
        hs = self.graph_linear_self(h)
        # --- relational feature, from neighbor connection
        m = self.graph_linear_edge(h)
        m = F.reshape(m, (batch_size, num_node, self.out_channel, self.num_edge_type))
        m = F.transpose(m, (0, 3, 1, 2)) # N x num_edge_type x n x F_out
        hr = F.matmul(adj, m) # adj[N x num_edge_type x n x n]
        hr = F.sum(hr, axis=1)
        return hr + hs


class AtomEmbedRGCNReadout(chainer.Chain):

    def __init__(self, in_channel, out_channel, nobias=True):
        super(AtomEmbedRGCNReadout, self).__init__()
        with self.init_scope():
            pass


class AtomEmbedRGCN(chainer.Chain):
    """Implementation for R-GCN used for atom embedding training
    """

    def __init__(self, word_size, num_atom_type=MAX_ATOMIC_NUM, num_edge_type=4,
                 ch_list=None, activation=F.relu):
        super(AtomEmbedRGCN, self).__init__()
        
        self.ch_list = [word_size]
        if not ch_list is None:
            self.ch_list.extend(ch_list)
        self.ch_list.append(num_atom_type)
        self.activation = activation
