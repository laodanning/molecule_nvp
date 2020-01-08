import chainer
import chainer.links as L
import chainer.functions as F

from model.atom_embed.atom_embed import atom_embed_model
from model.hyperparameter import Hyperparameter
from model.nvp_model.coupling import AffineAdjCoupling, AdditiveAdjCoupling, \
    AffineNodeFeatureCoupling, AdditiveNodeFeatureCoupling, RandomCouplingBlock
from model.nvp_model.mlp import BasicMLP
from data.utils import *

import math
import os
import logging as log

class MoleculeNVPModel(chainer.Chain):
    def __init__(self, hyperparams):
        super(MoleculeNVPModel, self).__init__()
        self.hyperparams = hyperparams
        self.masks = dict()
        self.masks["relation"] = self._create_masks("relation")
        self.masks["feature"] = self._create_masks("feature")
        assert self.hyperparams.gnn_type in ["relgcn", "gat"]
        self.add_self_loop = self.hyperparams.gnn_type == "gat"
        self.adj_size = self.hyperparams.num_nodes * \
            self.hyperparams.num_nodes * self.hyperparams.num_edge_types
        self.x_size = self.hyperparams.num_nodes * self.hyperparams.num_features
        self.max_atom_types = self.hyperparams.max_atom_types

        with self.init_scope():
            initial_ln_z_var = math.log(self.hyperparams.initial_z_var)
            if self.hyperparams.learn_dist:
                self.x_ln_var = chainer.Parameter(initializer=initial_ln_z_var, shape=[1])
                self.adj_ln_var = chainer.Parameter(initializer=initial_ln_z_var, shape=[1])
            else:
                self.x_ln_var = chainer.Variable(initializer=initial_ln_z_var, shape=[1])
                self.adj_ln_var = chainer.Variable(initializer=initial_ln_z_var, shape=[1])

            feature_coupling = AdditiveNodeFeatureCoupling if self.hyperparams.additive_feature_coupling else AffineNodeFeatureCoupling
            relation_coupling = AdditiveAdjCoupling if self.hyperparams.additive_relation_coupling else AffineAdjCoupling
            clinks = [
                feature_coupling(self.hyperparams.num_nodes, self.hyperparams.num_edge_types, self.hyperparams.num_features,
                                 self.masks["feature"][i %
                                                       self.hyperparams.num_nodes],
                                 batch_norm=self.hyperparams.apply_batchnorm, ch_list=self.hyperparams.gnn_fc_channels,
                                 gnn_type=self.hyperparams.gnn_type, gnn_params=self.hyperparams.gnn_params)
                for i in range(self.hyperparams.num_coupling["feature"])]
            clinks.extend([
                relation_coupling(self.hyperparams.num_nodes, self.hyperparams.num_edge_types, self.hyperparams.num_features,
                                  self.masks["relation"][i %
                                                         self.hyperparams.num_nodes],
                                  batch_norm=self.hyperparams.apply_batchnorm, ch_list=self.hyperparams.mlp_channels)
                for i in range(self.hyperparams.num_coupling["relation"])])
            self.clinks = chainer.ChainList(*clinks)

    def __call__(self, x, adj):
        # x (batch_size, ): atom id array
        # h (batch_size, max_atom_type): one-hot vec
        h = chainer.as_variable(x)
        choices = self.xp.eye(self.max_atom_types)
        h = choices[h.array]
        h = F.cast(h, self.xp.float32)

        # add noise
        if chainer.config.train:
            h += self.xp.random.uniform(0, 0.9, h.shape)

        adj = chainer.as_variable(adj)
        sum_log_det_jacobian_x = chainer.as_variable(
            self.xp.zeros([h.shape[0]], dtype=self.xp.float32))
        sum_log_det_jacobian_adj = chainer.as_variable(
            self.xp.zeros([h.shape[0]], dtype=self.xp.float32))

        # Input adj DOES NOT have self loop, we add self loop here for computation.
        if self.add_self_loop:
            adj_forward = adj + self.xp.eye(self.hyperparams.num_nodes)
        else:
            adj_forward = adj

        # forward step for channel-coupling layers
        for i in range(self.hyperparams.num_coupling["feature"]):
            h, log_det_jacobians = self.clinks[i](h, adj_forward)
            sum_log_det_jacobian_x += log_det_jacobians

        # add uniform noise to adjacency tensors
        if chainer.config.train:
            adj += self.xp.random.uniform(0, 0.9, adj.shape)

        # forward step for adjacency-coupling layers
        num_prev = self.hyperparams.num_coupling["feature"] + self.hyperparams.num_coupling["relation"]
        for i in range(self.hyperparams.num_coupling["feature"], num_prev):
            adj, log_det_jacobians = self.clinks[i](adj)
            sum_log_det_jacobian_adj += log_det_jacobians

        adj = F.reshape(adj, (adj.shape[0], -1))
        h = F.reshape(h, (h.shape[0], -1))
        out = [h, adj]
        return out, [sum_log_det_jacobian_x, sum_log_det_jacobian_adj]

    def reverse(self, z, true_adj=None, norm_sample=True):
        """
        Returns a molecule, given its latent vector.
        :param z: latent vector. Shape: [B, N*N*M + N*T]
            B = Batch size, N = number of atoms, M = number of bond types,
            T = number of atom types (Carbon, Oxygen etc.)
        :param true_adj: used for testing. An adjacency matrix of a real molecule
        :return: adjacency matrix and feature matrix of a molecule
        """
        batch_size = z.shape[0]
        with chainer.no_backprop_mode():
            z_x, z_adj = F.split_axis(chainer.as_variable(z), [self.x_size], 1)
            h_x = F.reshape(z_x, (batch_size, self.hyperparams.num_nodes, self.hyperparams.num_features))
            h_adj = F.reshape(z_adj, (batch_size, self.hyperparams.num_edge_types, self.hyperparams.num_nodes, self.hyperparams.num_nodes))

            if true_adj is None:
                # the adjacency coupling layers are applied in reverse order to get h_adj
                for i in reversed(range(self.hyperparams.num_coupling["feature"], len(self.clinks))):
                    h_adj, _ = self.clinks[i].reverse(h_adj)

                # make adjacency matrix from h_adj
                # 1. make it symmetric
                adj = h_adj + self.xp.transpose(h_adj, (0, 1, 3, 2))
                adj = adj / 2
                # 2. apply normalization along edge type axis and choose the most likely edge type.
                adj = F.softmax(adj, axis=1)
                max_bond = F.repeat(F.max(adj, axis=1).reshape(batch_size, -1, self.hyperparams.num_nodes, self.hyperparams.num_nodes),
                                    self.hyperparams.num_edge_types, axis=1)
                adj = F.floor(adj / max_bond)
                adj *= (1 - self.xp.eye(self.hyperparams.num_nodes)) # remove wrong self-loop    
            else:
                adj = true_adj
            
            adj_forward = adj + self.xp.eye(self.hyperparams.num_nodes) if self.add_self_loop else adj

            # feature coupling layers
            for i in reversed(range(self.hyperparams.num_coupling["feature"])):
                h_x, _ = self.clinks[i].reverse(h_x, adj_forward)

        atom_ids = F.argmax(h_x, axis=2)
        return atom_ids, adj

    def log_prob(self, z, log_det_jacobians):
        adj_ln_var = self.adj_ln_var * self.xp.ones([self.adj_size])
        x_ln_var = self.x_ln_var * self.xp.ones([self.x_size])
        log_det_jacobians[0] = log_det_jacobians[0] - self.x_size
        log_det_jacobians[1] = log_det_jacobians[1] - self.adj_size

        negative_log_likelihood_adj = F.average(F.sum(F.gaussian_nll(z[1], self.xp.zeros(
            self.adj_size, dtype=self.xp.float32), adj_ln_var, reduce="no"), axis=1) - log_det_jacobians[1])
        negative_log_likelihood_x = F.average(F.sum(F.gaussian_nll(z[0], self.xp.zeros(
            self.x_size, dtype=self.xp.float32), x_ln_var, reduce="no"), axis=1) - log_det_jacobians[0])

        negative_log_likelihood_adj /= self.adj_size
        negative_log_likelihood_x /= self.x_size

        if negative_log_likelihood_x.array < 0:
            log.warning("negative nll for x!")

        return [negative_log_likelihood_x, negative_log_likelihood_adj]

    def _create_masks(self, channel):
        if channel == "relation":  # for adjacenecy matrix
            return self._simple_masks(self.hyperparams.num_nodes)
        elif channel == "feature":  # for feature matrix
            return self._simple_masks(self.hyperparams.num_nodes)

    def _simple_masks(self, N):
        return ~self.xp.eye(N, dtype=self.xp.bool)

    def save_hyperparams(self, path):
        self.hyperparams.save(path)

    def load_hyperparams(self, path):
        self.hyperparams.load(path)

    def load_from(self, path):
        if os.path.exists(path):
            log.info("Try load model from {}".format(path))
            try:
                chainer.serializers.load_npz(path, self)
            except:
                log.warning("Fail in loading model from {}".format(path))
                return False
            return True
        raise ValueError("{} does not exist.".format(path))

    def input_from_smiles(self, smiles, atomic_num_list):
        return smiles_to_adj(smiles, self.hyperparams.num_nodes, self.hyperparams.num_nodes, atomic_num_list)

    @property
    def z_var(self):
        return [F.exp(self.x_ln_var).array[0], F.exp(self.adj_ln_var).array[0]]

    @property
    def ln_var(self):
        adj_ln_var = self.adj_ln_var * self.xp.ones([self.adj_size])
        x_ln_var = self.x_ln_var * self.xp.ones([self.x_size])
        return F.concat([x_ln_var, adj_ln_var], axis=0)
    
    @property
    def x_var(self):
        return F.exp(self.x_ln_var).array[0]
    
    @property
    def adj_var(self):
        return F.exp(self.adj_ln_var).array[0]
    
    @property
    def latent_size(self):
        return self.x_size + self.adj_size

    @property
    def learn_var(self):
        return self.hyperparams.learn_dist

    @property
    def mean(self):
        return self.xp.zeros(self.latent_size, dtype=self.xp.float32)

    def random_sample_latent(self, batch_size, temp=0.5):
        xp = self.xp
        z_dim = self.adj_size + self.x_size
        sigma_diag = xp.sqrt(xp.exp(self.ln_var.data)) * temp
        mu = xp.zeros([z_dim], dtype=xp.float32)
        return xp.random.normal(mu, sigma_diag, (batch_size, z_dim)).astype(xp.float32)

    def to_gpu(self, device=None):
        self.masks["relation"] = chainer.backends.cuda.to_gpu(
            self.masks["relation"], device=device)
        self.masks["feature"] = chainer.backends.cuda.to_gpu(
            self.masks["feature"], device=device)
        for clink in self.clinks:
            clink.to_gpu(device=device)
        super().to_gpu(device=device)

    def to_cpu(self):
        self.masks["relation"] = chainer.backends.cuda.to_cpu(
            self.masks["relation"])
        self.masks["feature"] = chainer.backends.cuda.to_cpu(
            self.masks["feature"])
        for clink in self.clinks:
            clink.to_cpu()
        super().to_cpu()
    
    def save_embed(self):
        chainer.serializers.save_npz(self.hyperparams.embed_model_path, self.embed_model)
