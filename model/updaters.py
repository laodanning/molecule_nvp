import chainer
from chainer import training
import chainer.functions as F
from data.utils import molecule_id_converter
from chainer import cuda
import six
from chainer import function
from data.utils import generate_mols, check_validity_with_label
from model.utils import get_device_id
from functools import reduce


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

class LatentNVPUpdater(object):
    def __init__(self, model, optimizer, device=None, 
                 batch_size=None, atomic_num_list=None):
        # self.model: NVPModel, consisting of MoleculeNVP model and LatentNVP model
        # this updater only update LatentNVP part
        self.model = model
        self.optimizer = optimizer
        self.optimizer.setup(model.latent_nvp)
        self.batch_size = batch_size if batch_size is not None else 32
        self.atomic_num_list = atomic_num_list
        self.device = device
    
    def update(self):
        # first random sample from moleculeNVP's latent space
        molecule_nvp = self.model.molecule_nvp
        latent_nvp = self.model.latent_nvp
        latent_samples = molecule_nvp.random_sample_latent(self.batch_size)

        # next get labels
        x, adj = molecule_nvp.reverse(latent_samples)
        labels = check_validity_with_label(x, adj, self.atomic_num_list, self.device)
        pos_degree = self.model.xp.sum(labels) / labels.shape[0]

        z, sum_log_det_jacs = latent_nvp(latent_samples)
        nll = latent_nvp.nll(z, labels, sum_log_det_jacs)
        latent_nvp.cleargrads()
        nll.backward()
        self.optimizer.update()
        return {"latent_nll": float(nll.data), "positive_sample_proportion": float(pos_degree)}
      

class MoleculeNVPUpdater(training.StandardUpdater):
    def __init__(self, iterator, optimizer,
                 converter=molecule_id_converter,
                 device=None, loss_func=None,
                 loss_scale=None, auto_new_epoch=True,
                 two_step=True, h_nll_weight=1, atomic_num_list=None):
        super().__init__(
            iterator, optimizer, converter=converter,
            device=device, loss_func=loss_func,
            loss_scale=loss_scale, auto_new_epoch=auto_new_epoch)

        self.model = optimizer.target
        self.two_step = two_step
        self.h_nll_weight = h_nll_weight
        self.atomic_num_list = atomic_num_list

    def update_core(self):
        batch = self._iterators["main"].next()
        x, adj = self.converter(batch, self.device)
        batch_size = x.shape[0]
        z, sum_log_det_jacs = self.model(x, adj)
        optimizer = self._optimizers["main"]
        # negative log likelihood (nll_h, nll_adj)
        nll = self.model.log_prob(z, sum_log_det_jacs)

        if self.two_step:
            loss = self.h_nll_weight * nll[0] + nll[1]
            chainer.report({"neg_log_likelihood": loss,
                            "nll_x": nll[0], "nll_adj": nll[1], "zx_var": self.model.z_var[0], "zA_var": self.model.z_var[1]})
        else:
            loss = nll
            chainer.report({"neg_log_likelihood": loss, "zx_var": self.model.z_var[0], "zA_var": self.model.z_var[1]})

        #loss += F.square(F.exp(self.model.ln_var) + F.exp(-self.model.ln_var))
        chainer.report({"ln_det_x": self.model.xp.mean(sum_log_det_jacs[0].array), "ln_det_adj": self.model.xp.mean(sum_log_det_jacs[1].array)})

        self.model.cleargrads()
        loss.backward()
        optimizer.update()

        # reverse valid-training
        if False:
            xp = self.model.xp
            self.model.cleargrads()
            device_id = get_device_id(self.device)
            x, adj = generate_mols(self.model, batch_size=batch_size, device=device_id, no_bp=False)
            x = chainer.functions.cast(x, xp.float32)
            labels = check_validity_with_label(x, adj, self.atomic_num_list, device=device_id)
            loss = 1 - xp.average(labels)
            grads = 1.0 - labels
            chainer.reporter.report({"valid_loss": loss})
            adj.grad = xp.ones_like(adj.array) * grads.reshape(batch_size, *(len(adj.shape)-1)*[1]) / (1.0 * self.model.adj_size)
            x.grad = xp.ones_like(x.array) * grads.reshape(batch_size, *(len(x.shape)-1)*[1]) / (1.0 * self.model.x_size)
            adj.backward()
            x.backward()
            optimizer.update()

        if self.auto_new_epoch and self._iterators["main"].is_new_epoch:
            optimizer.new_epoch(auto=True)


class DataParallelMoleculeNVPUpdater(training.ParallelUpdater):
    def __init__(self, iterator, optimizer,
                 converter=molecule_id_converter,
                 models=None, devices=None,
                 loss_func=None, loss_scale=None,
                 auto_new_epoch=True, two_step=True,
                 h_nll_weight=1):
        super().__init__(iterator, optimizer, converter=converter, models=models, devices=devices,
                         loss_func=loss_func, loss_scale=loss_scale, auto_new_epoch=auto_new_epoch)
        self.two_step = two_step
        self.h_nll_weight = h_nll_weight

    def update_core(self):
        optimizer = self.get_optimizer("main")
        model_main = optimizer.target
        models_others = {k: v for k, v in self._models.items()
                         if v is not model_main}

        iterator = self.get_iterator("main")
        batch = iterator.next()

        # -- split the batch to sub-batches -- #
        n = len(self._models)
        in_arrays_lists = {}
        for i, key in enumerate(six.iterkeys(self._models)):
            in_arrays_lists[key] = self.converter(
                batch[i::n], self._devices[key])

        # for reducing memory
        for model in six.itervalues(self._models):
            model.cleargrads()

        losses = []
        for model_key, model in six.iteritems(self._models):
            x, adj = in_arrays_lists[model_key]

            with function.force_backprop_mode():
                with chainer.using_device(self._devices[model_key]):
                    z, sum_log_det_jacs = model(x, adj)
                    nll = model.log_prob(z, sum_log_det_jacs)

                    if self.two_step:
                        loss = self.h_nll_weight * nll[0] + nll[1]
                    else:
                        loss = nll
                    #loss += F.square(F.exp(model.ln_var) + F.exp(-model.ln_var))
            losses.append(loss)

        for model in six.itervalues(self._models):
            model.cleargrads()

        for loss in losses:
            loss.backward(loss_scale=self.loss_scale)

        for model in six.itervalues(models_others):
            model_main.addgrads(model)

        total_loss = 0.0
        for loss in losses:
            loss_in_cpu = F.copy(loss, -1)
            total_loss += loss_in_cpu
        average_losses = total_loss / len(losses)
        chainer.report({"neg_log_likelihood": average_losses, "zx_var": model_main.z_var[0], "zA_var": model_main.z_var[1]})

        optimizer.update()

        for model in six.itervalues(models_others):
            model.copyparams(model_main)

        if self.auto_new_epoch and iterator.is_new_epoch:
            optimizer.new_epoch(auto=True)
