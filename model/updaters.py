import chainer
from chainer import training
import chainer.functions as F
from data.utils import molecule_id_converter
from chainer import cuda

class AtomEmbedUpdater(training.StandardUpdater):
    def __init__(self, iterator, optimizer,
                 converter=molecule_id_converter, 
                 device=None, loss_func=None, 
                 loss_scale=None, auto_new_epoch=True):
        super().__init__(iterator, optimizer, converter=converter, 
                         device=device, loss_func=loss_func, 
                         loss_scale=loss_scale, auto_new_epoch=auto_new_epoch)
        self.model = optimizer.target

    def update_core(self):
        iterator = self._iterators["main"]
        optimizer = self._optimizers["main"]

        batch = iterator.next()
        xs, adjs = self.converter(batch, self.device)
        xp = cuda.get_array_module(xs)

        hs = self.model(xs, adjs)
        hs = F.reshape(hs, (-1, hs.shape[-1]))
        xs = xs.reshape(-1)
        loss = F.softmax_cross_entropy(hs, xs)
        acc = F.accuracy(hs, xs)

        chainer.report({"ce_loss": loss, "accuracy": acc})

        self.model.cleargrads()
        loss.backward()
        optimizer.update()
