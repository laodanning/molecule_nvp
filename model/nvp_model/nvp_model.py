import chainer
import chainer.links as L
import chainer.functions as F

if __name__ == "__main__":
    import numpy as np
    import os, sys
    sys.path.append(os.getcwd())
    from data.utils import check_reverse

from model.hyperparameter import Hyperparameter
from model.nvp_model.latent_nvp import LatentNVP
from model.nvp_model.molecule_nvp import MoleculeNVPModel


class NVPModel(chainer.Chain):
    def __init__(self, hyperparams: Hyperparameter):
        super().__init__()
        molecule_nvp_params = hyperparams.subparams("molecule_nvp_params")
        latent_nvp_params = hyperparams.subparams("latent_nvp_params")

        with self.init_scope():
            self.molecule_nvp = MoleculeNVPModel(molecule_nvp_params)
            self.latent_nvp = LatentNVP(latent_nvp_params)

    def __call__(self, x, adj):
        out, _ = self.molecule_nvp(x, adj)
        z = F.concat(out, axis=1)
        z, _ = self.latent_nvp(z)
        return z

    def reverse(self, z, true_adj=None):
        z = self.latent_nvp.reverse(z)
        x, adj = self.molecule_nvp.reverse(z, true_adj)
        return x, adj
    
    @property
    def latent_size(self):
        return self.latent_nvp.latent_size
    
    @property
    def learn_var(self):
        return self.latent_nvp.learn_var

    @property
    def ln_var(self):
        return self.latent_nvp.ln_var
    
    @property
    def mean(self):
        return self.latent_nvp.mean

if __name__ == "__main__":
    batch_size = 10
    num_atoms = 9
    num_features = 8
    num_edge_types = 4
    params_dict = {
        "latent_nvp_params": {
            "channel_size": 396,
            "ch_list": [512, 128],
            "learn_var": True,
            "num_coupling_layers": 2
        },
        "molecule_nvp_params": {
            "max_atom_types": 8,
            "embed_model_path": "./output/qm9/final_embed_model.npz",
            "embed_model_hyper": "./output/qm9/atom_embed_model_hyper.json",
            "num_edge_types": num_edge_types,
            "num_features": num_features,
            "num_nodes": num_atoms,
            "apply_batchnorm": True,
            "num_coupling":
            {
                "feature": 4,
                "relation": 4
            },
            "gnn_type": "relgcn",
            "gnn_params": {
                "ch_list": [16, 64],
                "scale_adj": True
            },
            "gnn_fc_channels": [128, 64],
            "learn_dist": True,
            "mlp_channels": [256, 256],
            "additive_feature_coupling": False,
            "additive_relation_coupling": False,
            "feature_noise_scale": 0.0,
            "initial_z_var": 1.0,
            "apply_shuffle": True,
            "num_shuffle_blocks": 0,
            "block_params":
            {
                "num_x_couplings": 2,
                "num_A_couplings": 2,
                "x_ch_list": [128, 64],
                "A_ch_list": [256, 256]
            }
        }
    }
    hyper = Hyperparameter()
    hyper._parse_dict(params_dict)
    NVP = NVPModel(hyper)

    x = np.random.randint(0, 5, size=(batch_size, num_atoms))
    adj = np.random.randn(batch_size, num_edge_types, num_atoms, num_atoms).astype(np.float32)
    adj = adj + np.transpose(adj, (0, 1, 3, 2))
    adj /= 2
    adj = F.softmax(adj, axis=1)
    max_bond = F.repeat(F.max(adj, axis=1).reshape(batch_size, -1, num_atoms, num_atoms), num_edge_types, axis=1)
    adj = F.floor(adj / max_bond)
    adj *= (1 - np.eye(num_atoms))
    # # print(x.shape)
    # z = NVP(x, adj)
    # x_r, adj_r = NVP.reverse(z)


    z, _ = NVP.molecule_nvp(x, adj)
    z = F.concat(z, axis=1)
    x_r, adj_r = NVP.molecule_nvp.reverse(z)

    # check reverse
    print("reverse check for x: {}".format(check_reverse(x.astype(np.float32), F.cast(x_r, np.float32))))
    print("reverse check for adj: {}".format(check_reverse(adj, adj_r)))

    # zz, _ = NVP.latent_nvp(z)
    # zr = NVP.latent_nvp.reverse(zz)
    # print("reverse check for z: {}".format(check_reverse(z, zr)))
