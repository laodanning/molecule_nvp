import chainer
from chainer import cuda
import chainer.functions as F
from chainer_chemistry import MAX_ATOMIC_NUM
from chainer_chemistry.links import EmbedAtomID
from chainer_chemistry.links import GraphLinear
from model.hyperparameter import Hyperparameter
import json
import os

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
        x = F.cast(x, "int")
        if self.id_trans_fn is None:
            words = self.embed(x)
        else:
            words = self.embed(self.id_trans_fn(x))
        return words

    def atom_id(self, x, cos_dist=False):
        # x size: (batch_size, num_atom, word_size)
        # W siez: (num_atom_types, word_size)
        if cos_dist:
            return self.cosine_dist(x)
        else:
            return self.euclidean_dist(x)

    def euclidean_dist(self, x):
        word_matrix = self.embed.W
        x_shape = x.shape
        w_shape = word_matrix.shape
        compute_x = F.broadcast_to(F.reshape(x, (x_shape[0], x_shape[1], 1, x_shape[2])), (x_shape[0], x_shape[1], w_shape[0], x_shape[2]))
        diff = F.sum(F.square(compute_x - word_matrix), axis=-1)
        return F.argmin(diff, axis=-1)

    def cosine_dist(self, x):
        # x size: (batch_size, num_atom, word_size)
        # W siez: (num_atom_types, word_size)
        word_matrix = self.embed.W
        inner_prod_matrix = F.matmul(x, word_matrix.T)
        word_norms = self.xp.linalg.norm(word_matrix.array, axis=-1)
        x_norms = self.xp.linalg.norm(x.array, axis=-1)
        norms = self.xp.matmul(x_norms.reshape(x.shape[0], x.shape[1], 1), word_norms.reshape(1, self.num_atom_type))
        temp = inner_prod_matrix / norms
        temp2 = F.argmax(temp, axis=-1)
        return F.argmax(inner_prod_matrix / norms, axis=-1)

    def get_word_channel_stds(self):
        words = self.embed.W.array # (num_atom_types, word_size)
        # mean_square_w = self.xp.mean(words ** 2, axis=0)
        # square_mean_w = self.xp.mean(words, axis=0) ** 2
        # return mean_square_w - square_mean_w
        return self.xp.std(words, axis=0)


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
        # hs = self.graph_linear_self(h)
        # --- relational feature, from neighbor connection
        m = self.graph_linear_edge(h)
        m = F.reshape(m, (batch_size, num_node, self.out_channel, self.num_edge_type))
        m = F.transpose(m, (0, 3, 1, 2)) # N x num_edge_type x n x F_out
        hr = F.matmul(adj, m) # adj[N x num_edge_type x n x n]
        hr = F.sum(hr, axis=1)
        return hr


class AtomEmbedRGCN(chainer.Chain):
    """Implementation for R-GCN used for atom embedding training
    """

    def __init__(self, word_size, num_atom_type=MAX_ATOMIC_NUM, num_edge_type=4,
                 ch_list=None, scale_adj=False, activation=F.relu):
        super(AtomEmbedRGCN, self).__init__()
        
        self.ch_list = [word_size]
        if not ch_list is None:
            self.ch_list.extend(ch_list)
        # self.ch_list.append(num_atom_type)

        with self.init_scope():
            self.rgcn_convs = chainer.ChainList(*[
                AtomEmbedRGCNUpdate(self.ch_list[i], self.ch_list[i+1], num_edge_type) \
                for i in range(len(self.ch_list)-1)])
            self.rgcn_out = AtomEmbedRGCNUpdate(self.ch_list[-1], num_atom_type, num_edge_type)
        
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.activation = activation
        self.scale_adj = scale_adj

    def __call__(self, x, adj):
        """
        Args:
            x: (batchsize, num_nodes, word_size)
            adj: (batchsize, num_edge_type, num_node, num_nodes)

        Returns: (batchsize, num_nodes, num_atom_type)
        """
        h = x
        if self.scale_adj:
            adj = rescale_adj(adj)
        for rgcn_conv in self.rgcn_convs:
            h = self.activation(rgcn_conv(h, adj))
        h = self.rgcn_out(h, adj)
        return h
    
class AtomEmbedModel(chainer.Chain):
    def __init__(self, word_size, num_atom_type=MAX_ATOMIC_NUM, num_edge_type=4,
                 ch_list=None, scale_adj=False, activation=F.relu):
        super(AtomEmbedModel, self).__init__()
        with self.init_scope():
            self.embed = AtomEmbed(word_size, num_atom_type)
            self.rgcn = AtomEmbedRGCN(word_size, num_atom_type, 
                                      num_edge_type, ch_list, 
                                      scale_adj, activation)
        self.word_size = word_size
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.ch_list = ch_list
        self.scale_adj = scale_adj
    
    def __call__(self, x, adj):
        words = self.embed(x)
        h = self.rgcn(words, adj)
        return h

    def embedding(self, x):
        return self.embed(x)
    
    def atomid(self, words):
        return self.embed.atom_id(words)
    
    def word_channel_stds(self):
        return self.embed.get_word_channel_stds()

    def save_hyperparameters(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        hyper_params = {
            "word_size": self.word_size,
            "num_atom_type": self.num_atom_type,
            "num_edge_type": self.num_edge_type,
            "ch_list": self.ch_list,
            "scale_adj": self.scale_adj
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(hyper_params, f, indent=4, separators=(',', ': '))

def atom_embed_model(hyperparams: Hyperparameter) -> AtomEmbedModel:
    return AtomEmbedModel(
        hyperparams.word_size, 
        hyperparams.num_atom_type,
        hyperparams.num_edge_type,
        hyperparams.ch_list,
        hyperparams.scale_adj)


if __name__ == "__main__":
    import numpy as np

    batch_size = 5
    nodes = 9
    num_atom_type = 5
    num_edge_type = 4
    word_size = 8

    adj = np.random.randint(0, high=2, size=(batch_size, num_edge_type, nodes, nodes)).astype(np.float32)
    x = np.random.randint(0, high=num_atom_type, size=(batch_size, nodes))

    embed_layer = AtomEmbed(word_size, num_atom_type)
    rgcn = AtomEmbedRGCN(word_size, num_atom_type, num_edge_type, [8, 16], True)

    print("in size: ", x.shape, adj.shape)
    words = embed_layer(x)
    print("word size: ", words.shape)
    out = rgcn(words, adj)
    print("out size: ", out.shape)
