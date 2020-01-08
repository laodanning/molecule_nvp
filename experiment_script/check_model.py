import argparse
import os, sys
import numpy as np
import chainer
import logging as log
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.atom_embed import atom_embed
from model.hyperparameter import Hyperparameter
from model.nvp_model.molecule_nvp import MoleculeNVPModel
from data.utils import generate_mols, check_validity, get_atomic_num_id, \
    check_novelty, get_validation_idxs, adj_to_smiles, load_dataset
from chainer_chemistry.datasets import NumpyTupleDataset

if __name__ == "__main__":
    log.basicConfig(level=log.WARN)
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--params", type=str, default=None, help="path to model hyperparameters.")
    parser.add_argument("-m", "--model", type=str, default=None, help="path to model file.")
    args = parser.parse_args()
    global_params = Hyperparameter(args.params)
    model_params = global_params.subparams("model")
    config_params = global_params.subparams("configuration")
    dataset_params = global_params.subparams("dataset")

    # -- load dataset -- #
    if config_params.has("train_validation_split"):
        validation_idxs = get_validation_idxs(os.path.join(
            config_params.root_dir, config_params.train_validation_split))
    else:
        validation_idxs = None
    train_set, validation_set, dataset = load_dataset(os.path.join(
        dataset_params.root_dir, dataset_params.name), validation_idxs)
    atomic_num_ids = get_atomic_num_id(os.path.join(config_params.root_dir, config_params.atom_id_to_atomic_num))
    train_smiles = adj_to_smiles(train_set, atomic_num_ids)

    model = MoleculeNVPModel(model_params)
    model.load_from(args.model)
    device = 0
    model.to_gpu(device)
    chainer.backends.cuda.get_device_from_id(device).use()
    temperatures = np.linspace(0, 1, num=11, dtype=np.float32)
    valids = []
    uniques = []
    novels = []
    times = 5

    with chainer.no_backprop_mode(), chainer.using_config("train", False):
        for t in temperatures:
            temp_valid = []
            temp_unique = []
            temp_novel = []
            for _ in range(times):
                xs, adjs = generate_mols(model, temp=t, batch_size=1000, device=device)
                res = check_validity(xs, adjs, atomic_num_ids, device=device)
                temp_valid.append(res["valid_ratio"])
                temp_unique.append(res["unique_ratio"])
                temp_novel.append(check_novelty(
                    res["valid_smiles"], train_smiles))
            valids.append(sum(temp_valid) / times)
            uniques.append(sum(temp_unique) / times)
            novels.append(sum(temp_novel) / times)
            print("temp: {}\tvalid: {}\tunique: {}\tnovel: {}".format(t, valids[-1], uniques[-1], novels[-1]))
    
    plt.figure()
    plt.grid(True)
    plt.xlabel("temperature")
    plt.ylabel("%")
    plt.plot(temperatures, valids, marker="x", label="valid")
    plt.plot(temperatures, uniques, marker="x", label="unique")
    plt.plot(temperatures, novels, marker="x", label="novel")
    plt.legend()
    plt.show()
    plt.close()

    # for i in range(5):
    #     xs, adjs = generate_mols(model, 0.02, batch_size=100, device=device)
    #     res = check_validity(xs, adjs, atomic_num_ids, device=device)
