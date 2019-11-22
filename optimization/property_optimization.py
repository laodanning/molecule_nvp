import chainer
from chainer.training import extensions
import numpy as np
import json
import argparse
import logging as log
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.utils import *
from model.utils import *
from model.hyperparameter import Hyperparameter
from chainer.datasets import TupleDataset
from optimization.property_regression import Regression, PropertyRegression

def process_smiles(smiles):
    return smiles if is_valid_smiles(smiles) else None

def build_target(target_weight, target_set, target_stats):
    result = 0
    for t in target_weight:
        name = t["name"]
        weight = t["weight"]
        if name in target_set:
            new_item = (target_set[name] - target_stats[name]["mean"]) / target_stats[name]["std"]
            result = result + weight * new_item
        else:
            log.warning("Cannot find target '{}' in latent dataset, we will ignore it.".format(name))
    if isinstance(result, int):
        raise ValueError("No valid target is given")
    return result

def train_opt_model(config: Hyperparameter):
    # -- process hyperparams -- #
    train_config = config.subparams("train")
    output_config = config.subparams("output")
    os.makedirs(output_config.root, exist_ok=True)
    device = train_config.device
    batch_size = train_config.batch_size
    num_epoch = train_config.num_epoch
    log.info("training configurations:\n{}\n".format(config))

    # -- load latent dataset -- #
    train_set_path = config.latent_dataset["train"]
    val_set_path = config.latent_dataset["validation"]
    raw_train_set = pickle_load(train_set_path)
    raw_val_set = pickle_load(val_set_path)
    train_target_set = build_target(config.target, raw_train_set["targets"], raw_train_set["stats"])
    val_target_set = build_target(config.target, raw_val_set["targets"], raw_train_set["stats"])
    train_set = TupleDataset(raw_train_set["latent_vecs"], train_target_set)
    val_set = TupleDataset(raw_val_set["latent_vecs"], val_target_set)
    train_iter = chainer.iterators.SerialIterator(train_set, batch_size=batch_size, shuffle=True)
    val_iter = chainer.iterators.SerialIterator(val_set, batch_size=batch_size, shuffle=False, repeat=False)

    # -- build model -- #
    latent_size = raw_train_set["latent_vecs"][0].shape[-1]
    ch_list = config.model["ch_list"]
    model = PropertyRegression(latent_size, ch_list)
    reg_model = Regression(model)
    if device >= 0:
        reg_model.to_gpu(device)
    optimizer = chainer.optimizers.Adam(weight_decay_rate=train_config.weight_decay)
    optimizer = optimizer.setup(reg_model)
    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = chainer.training.Trainer(updater, (num_epoch, "epoch"), out=output_config.root)
    if train_config.has("save_epoch"):
        save_epoch = train_config.save_epoch
    else:
        save_epoch = None
    
    # -- trainer extension -- #
    if save_epoch is not None:
        trainer.extend(extensions.snapshot(), trigger=(save_epoch, "epoch"))
    trainer.extend(extensions.LogReport(filename=train_config.train_log_file))
    trainer.extend(extensions.Evaluator(val_iter, reg_model, device=device))
    trainer.extend(extensions.PrintReport(["epoch", "main/loss", "main/err", "validation/main/loss", "validation/main/err", "elapsed_time"]))
    trainer.extend(extensions.ProgressBar())

    trainer.run()
    chainer.serializers.save_npz(os.path.join(output_config.root, output_config.model_name), model)
    config.save(os.path.join(output_config.root, output_config.hyperparams))


def mol_property_optimization(config: Hyperparameter, input_smiles: str):
    chainer.config.train = False
    device = config.train["device"]
    if input_smiles is None:
        log.warning("No valid smiles is given, we will select one from validation set randomly.")

        # -- load dataset -- #
        val_set_path = config.latent_dataset["validation"]
        val_set = pickle_load(val_set_path)
        smiles = val_set["smiles"]
        idx = np.random.randint(0, len(smiles))
        input_smiles = smiles[idx]
        log.warning("Selected smiles: '{}'.".format(input_smiles))
    
    # -- configurations -- #
    output_config = config.subparams("output")
    model_path = os.path.join(output_config.root, output_config.model_name)
    atomic_num_list = get_atomic_num_id(config.atomic_num_list)

    # -- load nvp model -- #
    nvp_model_hyperparams = Hyperparameter(config.nvp_hyperparams).subparams("model")
    nvp_model = load_model_from(config.nvp_model_path, nvp_model_hyperparams)
    latent_size = nvp_model.adj_size + nvp_model.x_size

    # -- load regression model -- #
    model = PropertyRegression(latent_size, config.model["ch_list"])
    chainer.serializers.load_npz(model_path, model)

    # -- move model to device -- #
    if device >= 0:
        nvp_model.to_gpu(device)
        model.to_gpu(device)
        xp = chainer.backends.cuda.cupy
    else:
        xp = np

    # -- get mol input -- #
    input_x, input_adj = nvp_model.input_from_smiles(input_smiles, atomic_num_list)
    input_x = xp.expand_dims(input_x, axis=0)
    input_adj = xp.expand_dims(input_adj, axis=0)
    if device >= 0:
        input_x = chainer.backends.cuda.to_gpu(input_x, device)
        input_adj = chainer.backends.cuda.to_gpu(input_adj, device)
    with chainer.no_backprop_mode():
        input_z = chainer.functions.hstack(nvp_model(input_x, input_adj)[0])
    print(input_z.shape)

    step_size = 0.01
    num_iter = 5
    mols = []
    for i in range(num_iter):
        input_z = model.adjust_input(input_z, step_size)
        

def property_optimization(args):
    # -- decode configurations -- #
    config = Hyperparameter(args.config)
    is_train = args.train
    input_smiles = process_smiles(args.mol_smiles)

    if is_train:
        log.info("Training model...")
        train_opt_model(config)
    else:
        log.info("Do mol optimization...")
        mol_property_optimization(config, input_smiles)

if __name__ == "__main__":
    log.basicConfig(level=log.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./config/mol_optimization_qm9.json", help="path to configuration file.")
    parser.add_argument("-t", "--train", action="store_true", help="training mode flag.")
    parser.add_argument("-m", "--mol_smiles", type=str, default=None, help="input molecule smiles string, default is None")
    args = parser.parse_args()
    property_optimization(args)
