import chainer
from chainer import training, function, cuda, configuration
from chainer import functions as F
from data.utils import molecule_id_converter
import warnings
import copy
from chainer import reporter as reporter_module

class AtomEmbedEvaluator(training.extensions.Evaluator):
    def __init__(self, iterator, model, reporter,
                 converter=molecule_id_converter, 
                 device=None, eval_hook=None, name="validation"):
        super().__init__(iterator, model, 
                         converter=converter, 
                         device=device, 
                         eval_hook=eval_hook)
        self.model = model
        self.name = name
        self.reporter = reporter
        self.reporter.add_observer(name, self)
    
    def __call__(self, trainer=None):
        with configuration.using_config("train", False):
            result = self.evaluate()
        self.reporter.report(result, self)
        
        return result

    def evaluate(self):
        iterator = self._iterators["main"]
        
        if self.eval_hook:
            self.eval_hook(self)
        
        if hasattr(iterator, "reset"):
            iterator.reset()
            it = iterator
        else:
            warnings.warn('This iterator does not have the reset method. Evaluator '
                          'copies the iterator instead of resetting. This behavior is '
                          'deprecated. Please implement the reset method.',
                          DeprecationWarning)
            it = copy.copy(iterator)
        
        summary = reporter_module.DictSummary()

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                xs, adjs = self.converter(batch, self.device)
                xp = cuda.get_array_module(xs)
                with chainer.no_backprop_mode(), chainer.using_config("train", False):
                    hs = self.model(xs, adjs)
                    hs = F.reshape(hs, (-1, hs.shape[-1]))
                    xs = xs.reshape(-1)
                    loss = F.softmax_cross_entropy(hs, xs)
                    acc = F.accuracy(hs, xs)
                reporter_module.report({"accuracy": acc, "ce_loss": loss})
            summary.add(observation)

        return summary.compute_mean()
