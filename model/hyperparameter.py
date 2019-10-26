import json
import os
import tabulate
import logging as log


class Hyperparameter(object):
    def __init__(self, path=None):
        self.path = path
        if path is not None:
            self.parse()

    def parse(self):
        if self.path is None or not os.path.exists(self.path):
            log.warning(
                "Cannot find model configuration file (json) in {}".format(self.path))
            return None

        with open(self.path, encoding="utf-8") as f:
            contents = json.load(f)

        for k, v in contents.items():
            setattr(self, k, v)

    def load(self, path):
        self.path = path
        self.parse()

    def save(self, path, change_path=False):
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)
        if change_path:
            self.path = path

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True)

    def print(self):
        rows = []
        for k, v in self.__dict__.items():
            rows.append([k, v])
        print(tabulate(rows))
    
    def has(self, attr_name):
        return hasattr(self, attr_name)

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
