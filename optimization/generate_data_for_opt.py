import argparse
import logging as log
import os
import sys
import pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chainer
import networkx as nx
import numpy as np
from rdkit.Chem import (QED, Crippen, Descriptors, Draw, MolFromSmiles,
                        MolToSmiles, rdmolops)
from tqdm import tqdm

import optimization.sascorer as sascorer
from data.utils import (adj_to_smiles, get_atomic_num_id, get_validation_idxs,
                        load_dataset, molecule_id_converter, construct_mol, pickle_save)
from model.hyperparameter import Hyperparameter
from model.utils import load_model_from
from chainer_chemistry.datasets import NumpyTupleDataset


def process_dataset(dataset, model, atomic_num_list, batch_size, device):
    # -- model hyperparams -- #
    model_params = model.hyperparams

    # -- process dataset -- #
    log.info("Processing dataset...")
    valid_smiles = []
    valid_mols = []
    valid_xs = []
    valid_adjs = []
    for x, adj in dataset:
        smiles = MolToSmiles(construct_mol(x, adj, atomic_num_list), isomericSmiles=False, canonical=True)
        mol = MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles.append(smiles)
            valid_mols.append(mol)
            valid_xs.append(x)
            valid_adjs.append(adj)
    valid_xs = np.array(valid_xs)
    valid_adjs = np.array(valid_adjs)

    logP_values = []
    QED_values = []
    SA_scores = []
    cycle_scores = []

    log.info("Getting target values...")
    for mol in tqdm(valid_mols):
        logP_values.append(Crippen.MolLogP(mol, includeHs=True) if mol is not None else None)
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

    logP_stats = {"mean": np.mean(logP_values), "std": np.std(logP_values)}
    QED_stats = {"mean": np.mean(QED_values), "std": np.std(QED_values)}
    SA_scores_stats = {"mean": np.mean(SA_scores), "std": np.std(SA_scores)}
    cycle_scores_state = {"mean": np.mean(cycle_scores), "std": np.std(cycle_scores)}

    valid_set = NumpyTupleDataset(valid_xs, valid_adjs)
    batch_iter = chainer.iterators.SerialIterator(
        valid_set, batch_size, shuffle=False, repeat=False)

    # -- latent vectors -- #
    log.info("Generating latent vectors...")
    xp = chainer.backends.cuda.cupy if device >= 0 else np
    x_size = model_params.num_features * model_params.num_nodes
    adj_size = model_params.num_edge_types * model_params.num_nodes * model_params.num_nodes
    latent_size = x_size + adj_size
    num_instance = len(valid_set)
    latent_vecs = xp.zeros([num_instance, latent_size], dtype=xp.float32)
    batch_count = 0
    num_batch = num_instance // batch_size + (num_instance % batch_size != 0)
    with chainer.no_backprop_mode():
        for batch in batch_iter:
            log.info("processing batch ({}/{})".format(batch_count+1, num_batch))
            xs, adjs = molecule_id_converter(batch, device)
            upper_bd = (batch_count + 1) * batch_size if batch_count + 1 < num_batch else num_instance
            zxs, zadjs = model(xs, adjs)[0]
            latent_vecs[batch_count * batch_size : upper_bd] = xp.hstack((zxs.data, zadjs.data))
            batch_count += 1
    
    if device >= 0:
        latent_vecs = chainer.backends.cuda.to_cpu(latent_vecs)

    opt_datas = {
        "smiles": valid_smiles,
        "latent_vecs": latent_vecs,
        "targets": {
            "logPs": logP_values,
            "QEDs": QED_values,
            "SA_scores": SA_scores,
            "cycle_scores": cycle_scores
        },
        "stats": {
            "logPs": logP_stats,
            "QEDs": QED_stats,
            "SA_scores": SA_scores_stats,
            "cycle_scores": cycle_scores_state
        }
    }
    return opt_datas

def generate(config: Hyperparameter, output_path: str) -> None:
    log.basicConfig(level=log.INFO)

    # -- hyperparams -- #
    dataset_params = config.subparams("dataset")
    config_params = config.subparams("configuration")
    model_params = config.subparams("model")
    model_res_params = config.subparams("output")
    train_params = config.subparams("train")
    os.makedirs(output_path, exist_ok=True)
    batch_size = train_params.batch_size
    chainer.config.train = False

    # -- first we load dataset -- #
    log.info("Loading datasets...")
    dataset_path = os.path.join(dataset_params.root_dir, dataset_params.name)
    if config_params.has("train_validation_split"):
        validation_idxs = get_validation_idxs(os.path.join(config_params.root_dir, config_params.train_validation_split))
    else:
        validation_idxs = None
    train_set, validation_set, dataset = load_dataset(dataset_path, validation_idxs)
    atomic_num_list = get_atomic_num_id(os.path.join(config_params.root_dir, config_params.atom_id_to_atomic_num))
    
    # -- build model -- #
    log.info("Building model...")
    model_path = os.path.join(model_res_params.root_dir, model_res_params.final_model_name)
    model = load_model_from(model_path, model_params)
    device = train_params.device
    if device >= 0:
        model.to_gpu(device)

    # -- process target and latent vectors -- #
    log.info("Processing training set")
    train_opt_data = process_dataset(train_set, model, atomic_num_list, batch_size, device)
    
    log.info("Processing validation set")
    validation_opt_data = process_dataset(validation_set, model, atomic_num_list, batch_size, device)

    # -- save -- #
    pickle_save(os.path.join(output_path, "mol_opt_data_train.pkl"), train_opt_data)
    pickle_save(os.path.join(output_path, "mol_opt_data_val.pkl"), validation_opt_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, default="./output/qm9/hyperparams_relgcn.json", help="path to coniguration file")
    parser.add_argument("-o", "--output_path", type=str, default="./dataset/data_for_opt/qm9", help="output root path")
    args = parser.parse_args()
    config = Hyperparameter(args.config_path)
    generate(config, args.output_path)
