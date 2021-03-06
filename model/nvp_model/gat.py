import chainer
import chainer.functions as F
import chainer.backends.cuda as cuda

from chainer_chemistry.links.readout.ggnn_readout import GGNNReadout
from chainer_chemistry.links.update.relgat_update import RelGATUpdate

class RelationalGAT(chainer.Chain):
    """Relational Graph Attention Networks (GAT)

    See: Veličković, Petar, et al. (2017).\
        Graph Attention Networks.\
        `arXiv:1701.10903 <https://arxiv.org/abs/1710.10903>`\
        Dan Busbridge, et al. (2018).\
        Relational Graph Attention Networks
        `<https://openreview.net/forum?id=Bklzkh0qFm>`\


    Args:
        out_dim (int): dimension of output feature vector
        hidden_dim (int): dimension of feature vector
            associated to each atom
        n_layers (int): number of layers
        n_heads (int): number of multi-head-attentions.
        n_edge_types (int): number of edge types.
        dropout_ratio (float): dropout ratio of the normalized attention
            coefficients
        negative_slope (float): LeakyRELU angle of the negative slope
        softmax_mode (str): take the softmax over the logits 'across' or
            'within' relation. If you would like to know the detail discussion,
            please refer Relational GAT paper.
        concat_hidden (bool): If set to True, readout is executed in each layer
            and the result is concatenated
        concat_heads (bool) : Whether to concat or average multi-head
            attentions
        weight_tying (bool): enable weight_tying or not

    """
    def __init__(self, out_dim, hidden_dim, n_heads=3, 
                 negative_slope=0.2, n_edge_types=4, n_layers=4, 
                 dropout_ratio=-1, activation=F.tanh,
                 softmax_mode="across", concat_hidden=False,
                 concat_heads=True, weight_tying=False):
        super(RelationalGAT, self).__init__()
        n_readout_layer = n_layers if concat_hidden else 1
        n_message_layer = n_layers
        with self.init_scope():
            update_layers = []
            for i in range(n_message_layer):
                if i > 0 and concat_heads:
                    input_dim = hidden_dim * n_heads
                else:
                    input_dim = hidden_dim
                update_layers.append(RelGATUpdate(
                    input_dim, hidden_dim, n_heads=n_heads,
                    n_edge_types=n_edge_types, 
                    dropout_ratio=dropout_ratio,
                    negative_slope=negative_slope,
                    softmax_mode=softmax_mode,
                    concat_heads=concat_heads))
            self.update_layers = chainer.ChainList(*update_layers)
            self.readout_layers = chainer.ChainList(*[GGNNReadout(
                out_dim=out_dim, in_channels=input_dim,
                activation=activation, activation_agg=activation) 
                for _ in range(n_readout_layer)])
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.concat_hidden = concat_hidden
        self.concat_heads = concat_heads
        self.weight_tying = weight_tying
        self.negative_slope = negative_slope
        self.n_edge_types = n_edge_types
        self.dropout_ratio = dropout_ratio
    
    def __call__(self, atom_array, adj):
        """Forward propagation

        Args:
            atom_array: minibatch of molecular which is
                represented with atom IDs (representing C, O, S, ...)
                `atom_array[mol_index, atom_index]` represents `mol_index`-th
                molecule's `atom_index`-th atomic number
            adj: minibatch of adjancency matrix with edge-type
                information

        Returns:
            ~chainer.Variable: minibatch of fingerprint
        """
        h = atom_array
        #h0 = F.copy(h, cuda.get_device_from_array(h.data).id)
        g_list = []
        for step in range(self.n_layers):
            message_layer_idx = 0 if self.weight_tying else step
            h = self.update_layers[message_layer_idx](h, adj)
            if self.concat_hidden:
                g = self.readout_layers[step](h)
                g_list.append(g)

        if self.concat_hidden:
            return F.concat(g_list, axis=1)
        else:
            g = self.readout_layers[0](h)
            return g
