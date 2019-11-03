import argparse
import json
import logging as log
import os

import chainer
from chainer import training
from chainer.training import extensions
from chainer_chemistry.datasets import NumpyTupleDataset

from data.utils import get_validation_idxs, generate_mols,\
     check_validity, get_atomic_num_id, save_mol_png
from model.atom_embed.atom_embed import AtomEmbedModel
from model.nvp_model.nvp_model import AttentionNvpModel
from model.updaters import NVPUpdater, DataParallelNVPUpdater
from model.evaluators import AtomEmbedEvaluator
from model.utils import get_and_log, get_optimizer, set_log_level, get_log_level
from model.hyperparameter import Hyperparameter


def train(hyperparams: Hyperparameter):
    # -- hyperparams -- #
    dataset_params = hyperparams.subparams("dataset")
    config_params = hyperparams.subparams("configuration")
    train_params = hyperparams.subparams("train")
    model_params = hyperparams.subparams("model")
    output_params = hyperparams.subparams("output")
    
    os.makedirs(output_params.root_dir, exist_ok=True)
    if hasattr(output_params, "logname"):
        log.basicConfig(filename=os.path.join(output_params.root_dir, output_params.logname),
                        filemode="w", level=get_log_level(output_params.log_level))
    else:
        log.basicConfig(level=get_log_level(output_params.log_level))
    hyperparams.save(os.path.join(output_params.root_dir, "hyperparams.json"))
    atomic_num_list = get_atomic_num_id(os.path.join(config_params.root_dir, config_params.atom_id_to_atomic_num))
    
    data_parallel = False
    if isinstance(train_params.device, int):
        main_device = train_params.device
        device = main_device
    elif isinstance(train_params.device, dict):
        main_device = train_params.device["main"]
        device = train_params.device
        data_parallel = True
    else:
        raise ValueError("Invalid device.")
    log.info("Main Device: {}".format(main_device))

    log.info("dataset hyperparameters:\n{}\n".format(dataset_params))
    log.info("configuration hyperparameters:\n{}\n".format(config_params))
    log.info("train hyperparameters:\n{}\n".format(train_params))
    log.info("model hyperparameters:\n{}\n".format(model_params))
    log.info("output hyperparameters:\n{}\n".format(output_params))

    # -- build dataset -- #
    if config_params.has("train_validation_split"):
        validation_idxs = get_validation_idxs(os.path.join(
            config_params.root_dir, config_params.train_validation_split))
    else:
        validation_idxs = None

    dataset = NumpyTupleDataset.load(os.path.join(
        dataset_params.root_dir, dataset_params.name))
    if validation_idxs:
        train_idxs = [i for i in range(
            len(dataset)) if i not in validation_idxs]
        trainset_size = len(train_idxs)
        train_idxs.extend(validation_idxs)
        trainset, valset = chainer.datasets.split_dataset(
            dataset, trainset_size, train_idxs)
    else:
        trainset, valset = chainer.datasets.split_dataset_random(
            dataset, int(len(dataset) * 0.8), seed=777)

    train_iter = chainer.iterators.SerialIterator(
        trainset, train_params.batch_size, shuffle=True)
    val_iter = chainer.iterators.SerialIterator(
        valset, train_params.batch_size, repeat=False, shuffle=False)

    # -- model -- #
    model = AttentionNvpModel(model_params)
    if isinstance(device, dict):
        log.info("Using multi-GPU {}".format(device))
        model.to_gpu(main_device)
    elif device >= 0:
        log.info("Using GPU {}".format(device))
        chainer.cuda.get_device(main_device).use()
        model.to_gpu(device)
    else:
        log.info("Using CPU")

    # -- training details -- #
    num_epoch = train_params.num_epoch
    opt_gen = get_optimizer(train_params.optimizer)
    if train_params.has("optimizer_params"):
        optimizer = opt_gen(**train_params.optimizer_params)
    else:
        optimizer = opt_gen()

    optimizer.setup(model)
    if data_parallel:
        updater = DataParallelNVPUpdater(
            train_iter,
            optimizer,
            devices=device,
            two_step=train_params.two_step,
            h_nll_weight=train_params.h_nll_weight)
    else:
        updater = NVPUpdater(
            train_iter,
            optimizer,
            device=device,
            two_step=train_params.two_step,
            h_nll_weight=train_params.h_nll_weight)
    trainer = training.Trainer(
        updater, (num_epoch, "epoch"), out=output_params.root_dir)
    if train_params.has("save_epoch"):
        save_epoch = train_params.save_epoch
    else:
        save_epoch = num_epoch

    # -- evaluation function -- #
    def print_validity(trainer):
        with chainer.using_device(chainer.backends.cuda.get_device_from_id(main_device)):
            save_mol = (get_log_level(output_params.log_level) <= log.DEBUG)
            x, adj = generate_mols(model, batch_size=100,
                                device=main_device)  # x: atom id
            valid_mols = check_validity(
                x, adj, atomic_num_list=atomic_num_list, device=main_device)
            if save_mol:
                mol_dir = os.path.join(output_params.root_dir, output_params.saved_mol_dir,
                                    "generated_{}".format(trainer.updater.epoch))
                os.makedirs(mol_dir, exist_ok=True)
                for i, mol in enumerate(valid_mols["valid_mols"]):
                    save_mol_png(mol, os.path.join(mol_dir, "{}.png".format(i)))

    # -- trainer extension -- #
    trainer.extend(extensions.snapshot(), trigger=(save_epoch, "epoch"))
    trainer.extend(extensions.LogReport(filename=output_params.trainlogname))
    trainer.extend(print_validity, trigger=(1, "epoch"))
    trainer.extend(extensions.PrintReport([
        "epoch", "neg_log_likelihood", "nll_x", "nll_adj", "z_var", "elapsed_time"]))
    trainer.extend(extensions.ProgressBar())
    
    # -- start train -- #
    if hasattr(train_params, "load_snapshot"):
        chainer.serializers.load_npz(train_params.load_snapshot, trainer)
    trainer.run()
    chainer.serializers.save_npz(os.path.join(output_params.root_dir, output_params.final_model_name), model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-P", "--params", type=str, default=None,
                        help="path to your hyperparameter file (json file).")
    args = parser.parse_args()
    hyperparams = Hyperparameter(args.params)
    train(hyperparams)
