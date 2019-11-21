import chainer
from chainer import function_node
import chainer.functions
from chainer.utils import type_check
from model.nvp_model.mlp import MLP
import numpy as np
import argparse


class PropertyRegression(chainer.Chain):
    """Molecule Property Regression Model

    This model maps latent vector to molecule
    property target.
    """
    def __init__(self, latent_size, ch_list=None):
        super().__init__()
        self.latent_size = latent_size
        if ch_list is None:
            ch_list = []
        self.ch_list = ch_list + [1]
        with self.init_scope():
            self.mlp = MLP(ch_list, latent_size)
        
    def __call__(self, z) -> chainer.Variable:
        return self.mlp(z)
    
    def adjust_input(self, x, step_size):
        h = x
        if not isinstance(x, chainer.Variable):
            h = chainer.Variable(h, requires_grad=True)
        
        with chainer.using_config("train", False):
            y = self.__call__(h)
            y.backward(retain_grad=True)
            grad = h.grad()
        
        direction = grad / self.xp.linalg.norm(grad, axis=-1, keepdims=True)
        return x + direction * step_size


class Regression(chainer.Chain):
    def __init__(self, model, loss_func=chainer.functions.mean_squared_error):
        super().__init__()
        self.loss_func = loss_func
        self.y = None
        self.loss = None
        with self.init_scope():
            self.model = model

    def forward(self, x, t):
        self.y = self.model(x)
        self.loss = self.loss_func(self.y, t)
        acc = chainer.functions.mean(chainer.functions.absolute(self.y - t) / t)
        chainer.report({"loss": self.loss, "acc": acc}, self)
        return self.loss
