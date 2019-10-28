import argparse
import json
import logging as log
import os

import chainer
from chainer import training
from chainer.training import extensions
from chainer_chemistry.datasets import NumpyTupleDataset

from data.utils import get_validation_idxs
from model.atom_embed.atom_embed import AtomEmbed, AtomEmbedRGCN, AtomEmbedModel
from model.nvp_model.nvp_model import AttentionNvpModel
from model.updaters import NVPUpdater
from model.evaluators import AtomEmbedEvaluator
from model.utils import get_and_log, get_optimizer
from model.hyperparameter import Hyperparameter

def train(hyperparams: Hyperparameter):
    # -- hyperparams -- #
    dataset_params = hyperparams.subparams("dataset")
    config_params = hyperparams.subparams("configuration")
    train_params = hyperparams.subparams("train")
    model_params = hyperparams.subparams("model")
    output_params = hyperparams.subparams("output")
    log.info("dataset hyperparameters:\n{}\n".format(dataset_params))
    log.info("configuration hyperparameters:\n{}\n".format(config_params))
    log.info("train hyperparameters:\n{}\n".format(train_params))
    log.info("model hyperparameters:\n{}\n".format(model_params))
    log.info("output hyperparameters:\n{}\n".format(output_params))

    os.makedirs(output_params.root_dir, exist_ok=True)
    hyperparams.save(os.path.join(output_params.root_dir, "hyperparams.json"))

    # -- build dataset -- #
    if config_params.has("train_validation_split"):
        validation_idxs = get_validation_idxs(os.path.join(config_params.root_dir, config_params.train_validation_split))
    else:
        validation_idxs = None

    dataset = NumpyTupleDataset.load(os.path.join(dataset_params.root_dir, dataset_params.name))
    if validation_idxs:
        train_idxs = [i for i in range(len(dataset)) if i not in validation_idxs]
        trainset_size = len(train_idxs)
        train_idxs.extend(validation_idxs)
        trainset, valset = chainer.datasets.split_dataset(dataset, trainset_size, train_idxs)
    else:
        trainset, valset = chainer.datasets.split_dataset_random(dataset, int(len(dataset) * 0.8), seed=777)

    train_iter = chainer.iterators.SerialIterator(trainset, train_params.batch_size, shuffle=True)
    val_iter = chainer.iterators.SerialIterator(valset, train_params.batch_size, repeat=False, shuffle=False)

    # -- model -- #
    model = AttentionNvpModel(model_params)

    # -- training details -- #
    device = train_params.device
    num_epoch = train_params.num_epoch
    if device >= 0:
        log.info("Using GPU")
        chainer.cuda.get_device(device).use()
        model.to_gpu(device)
    
    opt_gen = get_optimizer(train_params.optimizer)
    if train_params.has("optimizer_params"):
        optimizer = opt_gen(**train_params.optimizer_params)
    else:
        optimizer = opt_gen()
    
    optimizer.setup(model)
    updater = NVPUpdater(
        train_iter, 
        optimizer, 
        device=device, 
        two_step=train_params.two_step, 
        adj_nll_weight=train_params.adj_nll_weight)
    trainer = training.Trainer(updater, (num_epoch, "epoch"), out=output_params.root_dir)
    if train_params.has("save_epoch"):
        save_epoch = train_params.save_epoch
    else:
        save_epoch = num_epoch
    
    # -- trainer extension -- #
    trainer.extend(extensions.snapshot, trigger=(save_epoch, "epoch"))
    trainer.extend(extensions.LogReport(filename=output_params.logname))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--params", type=str, default=None, help="path to your hyperparameter file (json file).")
    args = parser.parse_args()
    hyperparams = Hyperparameter(args.params)
    train(hyperparams)
