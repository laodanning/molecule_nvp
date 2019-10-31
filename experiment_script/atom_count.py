import sys, os
sys.path.append(os.getcwd())

from chainer_chemistry.datasets import NumpyTupleDataset
from data.utils import get_atomic_num_id
import argparse
import pandas as pd
from tqdm import tqdm

periodic_table_path = "./config/elementlist.csv"
atomic_num_list_path = "./config/atomic_num_qm9.json"

def load_periodic_table(path=periodic_table_path):
    return pd.read_csv(path, names=["atomic_id", "symbol", "name"], index_col=0)

def load_id_to_atomic_num(path):
    return get_atomic_num_id(path)

if __name__ == "__main__":
    # -- get dataset -- #
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default=None, help="path to your dataset")
    parser.add_argument("-a", "--atomic_num_path", type=str, default=atomic_num_list_path, help="path to your atomic num list file")

    args = parser.parse_args()
    dataset = NumpyTupleDataset.load(args.path)
    atomic_num_list = load_id_to_atomic_num(args.atomic_num_path)

    periodic_table = load_periodic_table()
    result_dict = {}
    
    for x, _ in tqdm(dataset):
        atomic_ids = filter(lambda x: x > 0, map(lambda x: atomic_num_list[x], x.astype(int)))
        for a in atomic_ids:
            symbol = periodic_table.loc[a]["symbol"]
            result_dict[symbol] = result_dict[symbol] + 1 if symbol in result_dict else 1

    for k, v in result_dict.items():
        print("{:<5}{:<5}".format(k, v))
