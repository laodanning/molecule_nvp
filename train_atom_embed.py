import argparse
import json
import logging as log
import os

import chainer
from chainer_chemistry.datasets import NumpyTypleDataset

from data.utils import get_validation_idxs
from model.atom_embed.atom_embed import AtomEmbed, AtomEmbedRGCN, AtomEmbedModel
from model.utils import get_and_log, get_optimizer


def load_config(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    else:
        raise ValueError("Cannot find configuration file {}.".format(path))

def train(config):
    log.info("Hyper-parameters:")
    device = get_and_log(config, "device", -1)
    out_dir = get_and_log(config, "out_dir", "./output")
    config_dir = get_and_log(config, "config_dir", "./config")
    dataset_dir = get_and_log(config, "dataset_dir", "./dataset")
    validation_idxs_filepath = get_and_log(config, "train_validation_split")
    dataset_name = get_and_log(config, "dataset", required=True)
    atomic_nums = get_and_log(config, "atom_id_to_atomic_num", required=True)
    batch_size = get_and_log(config, "batch_size", required=True)
    num_epoch = get_and_log(config, "num_epoch", required=True)
    word_size = get_and_log(config, "embed_word_size", required=True)
    molecule_size = get_and_log(config, "molecule_size", required=True)
    num_atom_type = get_and_log(config, "num_atom_type", required=True)
    kekulized = get_and_log(config, "kekulize", False)
    layers = get_and_log(config, "layers", required=True)
    scale_adj = get_and_log(config, "scale_adj", True)
    log_path = get_and_log(config, "log", "stdout")
    optimizer_type = get_and_log(config, "optimizer", "adam")
    optimizer_params = get_and_log(config, "optimizer_params")
    num_edge_type = 4 if kekulized else 5

    validation_idxs = get_validation_idxs(validation_idxs_filepath)
    dataset = NumpyTypleDataset.load(os.path.join(dataset_dir, dataset_name))
    if validation_idxs:
        train_idxs = [i for i in range(len(dataset)) if i not in validation_idxs]
        trainset_size = len(train_idxs)
        train_idxs.extend(validation_idxs)
        trainset, testset = chainer.datasets.split_dataset(dataset, trainset_size, train_idxs)
    else:
        trainset, testset = chainer.datasets.split_dataset_random(dataset, int(len(dataset) * 0.8), seed=777)
    
    train_iter = chainer.iterators.SerialIterator(trainset, batch_size)
    
    model = AtomEmbedModel(word_size, num_atom_type, num_edge_type,
                           layers, scale_adj)
    model.save_hyperparameters(config)
    
    if device >= 0:
        log.info("Using GPU")
        chainer.cuda.get_device(device).use()
        model.to_gpu(device)

    model_save_dir = os.path.join(out_dir, "models")
    os.makedirs(model_save_dir, exist_ok=True)

    opt_func = get_optimizer()
    if optimizer_params is not None:
        optimizer = opt_func(optimizer_params)
    else:
        optimizer = opt_func()
    
    optimizer.setup(model)
    updater = chainer.iterators.SerialIterator()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, default=None, help="path to training configuration file.")
    args = parser.parse_args()
    train_config = load_config(args.config_path)
    train(train_config)
