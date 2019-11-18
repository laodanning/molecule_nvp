from chainer_chemistry.datasets import NumpyTupleDataset
import numpy as np

class NumpyTupleFilterDataset(NumpyTupleDataset):
    def __init__(self, *datasets):
        super().__init__(*datasets)
    
    def filter(self, cond):
        assert (len(cond) == self._length)
        self._length = sum(cond)
        for d in self._datasets:
            if isinstance(d, np.ndarray):
                d = d[cond]
            else:
                d = list(map(lambda x: x[-1], filter(lambda x: cond[x[0]], enumerate(d))))
        
