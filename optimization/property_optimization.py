import chainer
import numpy as np
import json
import argparse
import logging as log
import os

from data.utils import *
from model.utils import *
from model.hyperparameter import Hyperparameter
from chainer.datasets import TupleDataset
from optimization.property_regression import Regression, PropertyRegression

def process_smiles(smiles):
    return smiles if is_valid_smiles(smiles) else None

def build_target(target_set, target_weight, target_stats):
    result = 0
    for t in target_weight:
        name = t["name"]
        weight = t["weight"]
        if name in target_set:
            new_item = (target_set[name] - target_stats[name]["mean"]) / target_stats[name]["std"]
            result = result + weight * new_item
        else:
            log.warning("Connot find target '{}' in latent dataset, we will ignore it.".format(name))
    if result == 0:
        raise ValueError("No valid target is given")
    return result

def train_opt_model(config: Hyperparameter):
    # -- process hyperparams -- #
    train_config = config.subparams("train")
    output_config = config.subparams("output")
    os.makedirs(output_config.root, exist_ok=True)
    device = train_config.device

    # -- load latent dataset -- #
    train_set_path = config.latent_dataset["train"]
    val_set_path = config.latent_dataset["validation"]
    raw_train_set = np.load(train_set_path)
    raw_val_set = np.load(val_set_path)
    train_target_set = build_target(config.target, raw_train_set["targets"], raw_train_set["stats"])
    val_target_set = build_target(config.target, raw_val_set["targets"], raw_train_set["stats"])
    train_set = TupleDataset(raw_train_set["latent_vecs"], train_target_set)
    val_set = TupleDataset(raw_val_set["latent_vecs"], val_target_set)

    # -- build model -- #
    latent_size = raw_train_set["latent_vecs"][0].shape[-1]
    ch_list = config.model["ch_list"]
    model = PropertyRegression(latent_size, ch_list)
    reg_model = Regression(model)
    if device >= 0:
        reg_model.to_gpu(device)


def mol_property_optimization(config: Hyperparameter, input_smiles: str):
    pass


def property_optimization(args):
    # -- decode configurations -- #
    config = Hyperparameter(args.config)
    is_train = args.train
    input_smiles = process_smiles(args.mol_smiles)

    if is_train:
        train_opt_model(config)
    else:
        mol_property_optimization(config, input_smiles)

if __name__ == "__main__":
    log.basicConfig(level=log.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./config/mol_optimization_qm9.json", help="path to configuration file.")
    parser.add_argument("-t", "--train", action="store_true", help="training mode flag.")
    parser.add_argument("-m", "--mol_smiles", type=str, default=None, help="input molecule smiles string, default is None")
    args = parser.parse_args()
    property_optimization(args)
