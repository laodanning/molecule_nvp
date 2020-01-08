import argparse
import os, sys
import logging as log
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chainer
import numpy as np
from rdkit.Chem import Draw
from chainer_chemistry.datasets import NumpyTupleDataset

from data.utils import *
from model.hyperparameter import Hyperparameter
from model.nvp_model.molecule_nvp import MoleculeNVPModel
from model.utils import load_model_from
import pandas as pd

periodic_table_path = "./config/elementlist.csv"

def load_periodic_table(path=periodic_table_path):
    return pd.read_csv(path, names=["atomic_id", "symbol", "name"], index_col=0)

if __name__ == "__main__":
    log.basicConfig(level=log.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        default="./config/generate_qm9.json", help="path to configuration file")
    args = parser.parse_args()

    # -- hyperparams -- #
    gen_params = Hyperparameter(args.config)
    device = gen_params.device
    save_fig = gen_params.save_fig
    num_experiments = gen_params.num_experiments
    chainer.config.train = False

    # -- load model -- #
    periodic_table = load_periodic_table()
    model_hyperparams_path = os.path.join(
        gen_params.model_root_path, gen_params.model_hyperparams)
    model_path = os.path.join(
        gen_params.model_root_path, gen_params.model_file)
    model_params = Hyperparameter(model_hyperparams_path).subparams("model")
    model = load_model_from(model_path, model_params)
    if device >= 0:
        model.to_gpu(device)

    # -- load dataset -- #
    atomic_num_list = get_atomic_num_id(gen_params.atom_id_to_atomic_num)
    null_atom = atomic_num_list[-1]

    # -- random generation -- #
    x, adj = generate_mols(model, batch_size=1,
                            true_adj=None, temp=gen_params.temperature, device=device)
    val_result = check_validity(x, adj, atomic_num_list, device)
    if device >= 0:
        x.to_cpu()
        adj.to_cpu()

    x_str = map(lambda a: atomic_num_list[a], x.array[0])
    x_str = list(map(lambda a: periodic_table.loc[a]["symbol"] if a != null_atom else None, x_str))
    elems = list(map(lambda a: periodic_table.loc[a]["symbol"], atomic_num_list[:-1]))
    print(x_str)
    print(elems)
    # print(adj)

    if save_fig:

        img = Draw.MolsToGridImage(val_result["valid_mols"], legends=val_result["valid_smiles"],
                                    molsPerRow=1, subImgSize=(500, 500))
        img.show()
