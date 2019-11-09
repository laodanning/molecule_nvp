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
from model.updaters import AtomEmbedUpdater
from model.evaluators import AtomEmbedEvaluator
from model.utils import get_and_log, get_optimizer


def load_config(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    else:
        raise ValueError("Cannot find configuration file {}.".format(path))

def train(config):
    # -- read hyperparameters --
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
    save_epoch = get_and_log(config, "save_epoch", -1)
    kekulized = get_and_log(config, "kekulize", False)
    layers = get_and_log(config, "layers", required=True)
    scale_adj = get_and_log(config, "scale_adj", True)
    log_name = get_and_log(config, "log_name", "log")
    optimizer_type = get_and_log(config, "optimizer", "adam")
    optimizer_params = get_and_log(config, "optimizer_params")
    snapshot = get_and_log(config, "snapshot")
    num_edge_type = 4 if kekulized else 5
    
    os.makedirs(out_dir, exist_ok=True)

    if validation_idxs_filepath is not None:
        validation_idxs = get_validation_idxs(os.path.join(config_dir, validation_idxs_filepath))
    else:
        validation_idxs = None

    # -- build dataset --
    dataset = NumpyTupleDataset.load(os.path.join(dataset_dir, dataset_name))
    if validation_idxs:
        train_idxs = [i for i in range(len(dataset)) if i not in validation_idxs]
        trainset_size = len(train_idxs)
        train_idxs.extend(validation_idxs)
        trainset, testset = chainer.datasets.split_dataset(dataset, trainset_size, train_idxs)
    else:
        trainset, testset = chainer.datasets.split_dataset_random(dataset, int(len(dataset) * 0.8), seed=777)
    
    train_iter = chainer.iterators.SerialIterator(trainset, batch_size, shuffle=True)
    test_iter = chainer.iterators.SerialIterator(testset, batch_size, repeat=False, shuffle=False)
    
    # -- model --
    model = AtomEmbedModel(word_size, num_atom_type, num_edge_type,
                           layers, scale_adj)
    model.save_hyperparameters(os.path.join(out_dir, "atom_embed_model_hyper.json"))
    
    # -- training details --
    if device >= 0:
        log.info("Using GPU")
        chainer.cuda.get_device(device).use()
        model.to_gpu(device)

    opt_func = get_optimizer(optimizer_type)
    if optimizer_params is not None:
        optimizer = opt_func(optimizer_params)
    else:
        optimizer = opt_func()
    
    optimizer.setup(model)
    updater = AtomEmbedUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (num_epoch, "epoch"), out=out_dir)
    save_epoch = save_epoch if save_epoch >= 0 else num_epoch
    
    # -- trainer extension --
    trainer.extend(extensions.snapshot, trigger=(save_epoch, "epoch"))
    trainer.extend(extensions.LogReport(filename=log_name))
    trainer.extend(AtomEmbedEvaluator(test_iter, model, reporter=trainer.reporter, device=device))
    trainer.extend(extensions.PrintReport(["epoch", "ce_loss", "accuracy", "validation/ce_loss", "validation/accuracy", "elapsed_time"]))
    trainer.extend(extensions.PlotReport(["ce_loss", "validation/ce_loss"], x_key="epoch", filename="cross_entrypy_loss.png"))
    trainer.extend(extensions.PlotReport(["accuracy", "validation/accuracy"], x_key="epoch", filename="accuracy.png"))

    if snapshot is not None:
        chainer.serializers.load_npz(snapshot, trainer)
    trainer.run()
    chainer.serializers.save_npz(os.path.join(out_dir, "final_embed_model.npz"), model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, default=None, help="path to training configuration file.")
    args = parser.parse_args()
    train_config = load_config(args.config_path)
    train(train_config)
