import numpy as np
import json
import os
import logging as lg

def get_validation_idxs(file_path):
    lg.info("loading train/validation split information from {}".format(file_path))
    if os.path.exists(file_path):
        with open(file_path) as f:
            data = json.load(f)
    else:
        lg.warning("Cannot find validation index file in {}, divide dataset randomly.".format(file_path))
        data = None

    return data

def get_atomic_num_id(file_path):
    """Return a list which maps atom id to atomic number.
    atom id: id used during training.
    atomic number: e.g. C(6), O(8), H(1), etc. 0 for virtual atom.

    Args:
        file_path: path to json file which contains the list.
    Returns:
        A list which maps atom id to atomic number.
    """
    lg.info("loading information between atomic num and atom id "
            "from {}".format(file_path))
    if os.path.exists(file_path):
        with open(file_path) as f:
            data = json.load(f)
    else:
        raise ValueError("Cannot find atomid to atomic number information file in {}".format(file_path))
    return data

def trans_fn(data):
    pass
