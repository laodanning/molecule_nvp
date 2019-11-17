import argparse
import logging as log
import os
import sys

import chainer
import networkx as nx
import numpy as np
from rdkit.Chem import (QED, Crippen, Descriptors, Draw, MolFromSmiles,
                        MolToSmiles, rdmolops)
from tqdm import tqdm

import optimization.sascorer as sascorer
from data.utils import (adj_to_smiles, get_atomic_num_id, get_validation_idxs,
                        load_dataset, molecule_id_converter)
from model.hyperparameter import Hyperparameter
from model.utils import load_model_from

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate(config: Hyperparameter, output_path: str) -> None:
    log.basicConfig(level=log.INFO)

    # -- hyperparams -- #
    dataset_params = config.subparams("dataset")
    config_params = config.subparams("configuration")
    model_params = config.subparams("model")
    model_res_params = config.subparams("output")
    train_params = config.subparams("train")
    os.makedirs(output_path, exist_ok=True)

    # -- first we load dataset -- #
    log.info("Loading datasets...")
    dataset_path = os.path.join(dataset_params.root_dir, dataset_params.name)
    if config_params.has("train_validation_split"):
        validation_idxs = get_validation_idxs(os.path.join(config_params.root_dir, config_params.train_validation_split))
    else:
        validation_idxs = None
    train_set, validation_set, dataset = load_dataset(dataset_path, validation_idxs)
    atomic_num_list = get_atomic_num_id(os.path.join(config_params.root_dir, config_params.atom_id_to_atomic_num))
    train_smiles = adj_to_smiles(train_set, atomic_num_list)
    train_mols = [MolFromSmiles(m) for m in train_smiles]

    # -- build model -- #
    log.info("Building model...")
    model_path = os.path.join(model_res_params.root_dir, model_res_params.final_model_name)
    model = load_model_from(model_path, model_params)

    # -- targets -- #
    logP_values = []
    QED_values = []
    SA_scores = []
    cycle_scores = []

    log.info("Getting target values...")
    for mol in tqdm(train_mols):
        logP_values.append(Crippen.MolLogP(inMol=mol, addHs=True) if mol is not None else None)
        QED_values.append(QED.qed(mol) if mol is not None else None)
        SA_scores.append(-sascorer.calculateScore(mol) if mol is not None else None)
        
        cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([ len(j) for j in cycle_list ])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6
        cycle_scores.append(-cycle_length)
    
    logP_values = np.array(logP_values, dtype=np.float32)
    QED_values = np.array(QED_values, dtype=np.float32)
    SA_scores = np.array(SA_scores, dtype=np.float32)
    cycle_scores = np.array(cycle_scores, dtype=np.float32)

    logP_values_normalized = (logP_values - np.mean(logP_values)) / np.std(logP_values)
    QED_values_normalized = (QED_values - np.mean(QED_values)) / np.std(QED_values)
    SA_scores_normalized = (SA_scores - np.mean(SA_scores)) / np.std(SA_scores)
    cycle_scores_normalized = (cycle_scores - np.mean(cycle_scores)) / np.std(cycle_scores)

    batch_size = train_params.batch_size
    train_iter = chainer.iterators.SerialIterator(
        train_set, batch_size, shuffle=False, repeat=False)

    # -- latent vectors -- #
    log.info("Generating latent vectors...")
    device = train_params.device
    if device >= 0:
        model = model.to_gpu(device)
        xp = chainer.backends.cuda.cupy
    else:
        xp = np
    x_size = model_params.num_features * model_params.num_nodes
    adj_size = model_params.num_edge_types * model_params.num_nodes * model_params.num_nodes
    latent_size = x_size + adj_size
    num_train = len(train_set)
    latent_vecs = xp.zeros([num_train, latent_size], dtype=xp.float32)
    chainer.config.train = False
    batch_count = 0
    num_batch = num_train // batch_size + (num_train % batch_size != 0)
    with chainer.no_backprop_mode():
        for batch in train_iter:
            xs, adjs = molecule_id_converter(batch, device)
            upper_bd = (batch_count + 1) * batch_size if batch_count + 1 < num_batch else num_train
            zxs, zadjs = model(xs, adjs)[0]
            latent_vecs[batch_count * batch_size : upper_bd] = xp.hstack((zxs.data, zadjs.data))
    
    if device >= 0:
        latent_vecs = chainer.backends.cuda.to_cpu(latent_vecs)
    
    opt_datas = {
        "smiles": train_smiles,
        "latent_vecs": latent_vecs,
        "logPs": logP_values_normalized,
        "QEDs": QED_values_normalized,
        "SA_scales": SA_scores_normalized,
        "cycle_scores": cycle_scores_normalized
    }

    # -- save -- #
    np.savez(os.path.join(output_path, "mol_opt_data.npz"), opt_datas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, default="./output/qm9/hyperparams.json", help="path to coniguration file")
    parser.add_argument("-o", "--output_path", type=str, default="./dataset/data_for_opt/qm9", help="output root path")
    args = parser.parse_args()
    config = Hyperparameter(args.config_path)
    generate(config, args.output_path)
