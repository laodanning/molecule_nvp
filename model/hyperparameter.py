import json
import os
from tabulate import tabulate
import logging as log


class Hyperparameter(object):
    def __init__(self, path=None):
        self._path = path
        if path is not None:
            self.parse()

    def parse(self):
        if self._path is None or not os.path.exists(self._path):
            log.warning(
                "Cannot find model configuration file (json) in {}".format(self._path))
            return None

        with open(self._path, encoding="utf-8") as f:
            contents = json.load(f)

        self._parse_dict(contents)
    
    def _parse_dict(self, dict_obj):
        for k, v in dict_obj.items():
            setattr(self, k, v)

    def load(self, path):
        self._path = path
        self.parse()

    def save(self, path, change_path=False):
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        if change_path:
            self._path = path

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True)

    def printself(self):
        rows = []
        for k, v in self.__dict__.items():
            if k == "_path": continue
            rows.append([k, v])
        print(tabulate(rows))
    
    def __str__(self):
        rows = []
        for k, v in self.__dict__.items():
            if k == "_path": continue
            rows.append([k, v])
        return str(tabulate(rows))
        
    __repr__ = __str__
    
    def has(self, attr_name):
        return hasattr(self, attr_name)
    
    def subparams(self, domain):
        if hasattr(self, domain):
            sub_params = getattr(self, domain)
            if isinstance(sub_params, dict):
                result = Hyperparameter()
                result._parse_dict(sub_params)
                return result
        log.warning("Cannot find sub domain {} in {}".format(domain, __name__))
        return None
                

class NVPHyperParameter(Hyperparameter):
    def __init__(self, path=None):
        self.num_nodes = 9
        self.num_edge_types = 4
        self.num_features = 8
        self.learn_dist = True
        self.additive_feature_coupling = True
        self.additive_relation_coupling = False
        self.apply_batchnorm = True

        super().__init__(path=path)
