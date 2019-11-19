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
from model.nvp_model.nvp_model import MoleculeNVPModel
from model.utils import load_model_from

if __name__ == "__main__":
    log.basicConfig(level=log.DEBUG, filename="./test.log")
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
    validation_idxs = get_validation_idxs(gen_params.train_validation_split)
    dataset = NumpyTupleDataset.load(os.path.join(
        gen_params.data_root_path, gen_params.dataset))
    train_idxs = [i for i in range(len(dataset)) if i not in validation_idxs]
    n_train = len(train_idxs)
    train_idxs.extend(validation_idxs)
    train_set = chainer.datasets.SubDataset(
        dataset, 0, n_train, order=train_idxs)
    raw_smiles = adj_to_smiles(dataset, atomic_num_list)

    np.random.seed(1)
    mol_index = np.random.randint(0, len(dataset))
    x = np.expand_dims(dataset[mol_index][0], axis=0)
    adj = np.expand_dims(dataset[mol_index][1], axis=0)
    if device >= 0:
        adj = chainer.backends.cuda.to_gpu(adj, device)
        x = chainer.backends.cuda.to_gpu(x, device)
    z0 = model(x, adj)[0]
    if device >= 0:
        z0 = chainer.backends.cuda.cupy.hstack((z0[0].data, z0[1].data))
        adj = chainer.backends.cuda.to_cpu(adj)
        x = chainer.backends.cuda.to_cpu(x)
    else:
        z0 = np.hstack((z0[0].data, z0[1].data))
    rx, radj = model.reverse(z0, norm_sample=False)
    rx.to_cpu()
    radj.to_cpu()
    rx = rx.array[0]
    radj = radj.array[0]

    rsmiles = adj_to_smiles([(rx, radj)], atomic_num_list)
    smiles = raw_smiles[mol_index]
    print(smiles)
    print(rsmiles)
    print(np.array_equal(radj, adj))
    log.debug(adj)
    log.debug(radj)
    log.debug(np.equal(adj, radj))


