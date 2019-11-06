import argparse
import os
import logging as log

import chainer
import numpy as np
from rdkit.Chem import Draw
from chainer_chemistry.datasets import NumpyTupleDataset

from data.utils import adj_to_smiles, get_atomic_num_id, get_validation_idxs, \
    generate_mols, check_validity, check_novelty
from model.hyperparameter import Hyperparameter
from model.nvp_model.nvp_model import AttentionNvpModel
from model.utils import load_model_from

if __name__ == "__main__":
    log.basicConfig(level=log.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./config/generate_qm9.json", help="path to configuration file")
    args = parser.parse_args()

    # -- hyperparams -- #
    gen_params = Hyperparameter(args.config)
    device = gen_params.device
    save_fig = gen_params.save_fig
    num_experiments = gen_params.num_experiments
    chainer.config.train = False

    # -- load model -- #
    model_hyperparams_path = os.path.join(gen_params.model_root_path, gen_params.model_hyperparams)
    model_path = os.path.join(gen_params.model_root_path, gen_params.model_file)
    model_params = Hyperparameter(model_hyperparams_path)
    model = load_model_from(model_path, model_params)
    if device >= 0:
        model.to_gpu(device)

    # -- load dataset -- #
    atomic_num_list = get_atomic_num_id(gen_params.atom_id_to_atomic_num)
    validation_idxs = get_validation_idxs(gen_params.train_validation_split)
    dataset = NumpyTupleDataset.load(os.path.join(gen_params.data_root_path, gen_params.dataset))
    train_idxs = [i for i in range(len(dataset)) if i not in validation_idxs]
    n_train = len(train_idxs)
    train_idxs.extend(validation_idxs)
    train_set = chainer.datasets.SubDataset(dataset, 0, n_train, order=train_idxs)
    train_smiles = adj_to_smiles(train_set, atomic_num_list)

    # -- random generation -- #
    valid_ratio = []
    unique_ratio = []
    novel_ratio = []
    if save_fig:
        gen_dir = os.path.join(gen_params.output_root_path, "generated")
        os.makedirs(gen_dir, exist_ok=True)

    for i in range(num_experiments):
        x, adj = generate_mols(model, batch_size=gen_params.batch_size, true_adj=None, temp=gen_params.temperature, device=device)
        val_result = check_validity(x, adj, atomic_num_list, device)
        novel_ratio.append(check_novelty(val_result["valid_mols"], train_smiles))
        unique_ratio.append(val_result["unique_ratio"])
        valid_ratio.append(val_result["valid_ratio"])
        num_valid = len(val_result["valid_mols"])
    
        if save_fig:
            file_path = os.path.join(gen_dir, "generated_mols_{}.png".format(i))
            img = Draw.MolsToGridImage(val_result["valid_mols"], legends=val_result["valid_smiles"],
                                       molsPerRow=20, subImgSize=(300, 300))
            img.save(file_path)
        
    log.info("validity: mean={:.2f}%, sd={:.2f}%, vals={}".format(np.mean(valid_ratio), np.std(valid_ratio), valid_ratio))
    log.info("novelty: mean={:.2f}%, sd={:.2f}%, vals={}".format(np.mean(novel_ratio), np.std(novel_ratio), novel_ratio))
    log.info("uniqueness: mean={:.2f}%, sd={:.2f}%, vals={}".format(np.mean(unique_ratio), np.std(unique_ratio),
                                                                 unique_ratio))

    mol_smiles = None
    
    # -- Intepolation generation -- #
    if gen_params.draw_neighbor:
        for seed in range(5):
            filepath = os.path.join(gen_dir, "generated_interpolation_molecules_seed_{}.png".format(seed))
            log.info("saving {}".format(filepath))
            pass
        