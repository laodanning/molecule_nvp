import argparse
import os

from chainer.dataset import download
from chainer_chemistry.datasets import NumpyTupleDataset
from chainer_chemistry.dataset.preprocessors import RSGCNPreprocessor, GGNNPreprocessor
from data.relational_graph_preprocessor import RelationalGraphPreprocessor
from data.get_qm9 import get_qm9
from data.utils import get_atomic_num_id


def parse():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data_name", type=str, default="qm9",
                        choices=["qm9"],
                        help="dataset to be downloaded")
    parser.add_argument("--data_type", type=str, default="relgraph",
                        choices=["relgraph"],)
    parser.add_argument("--kekulize", action="store_true",
                        help="perform kekulization molecule or not")
    args = parser.parse_args()
    return args


args = parse()
data_name = args.data_name
data_type = args.data_type
kekulize = args.kekulize
print("args", vars(args))

if data_name == "qm9":
    max_atoms = 9
    id_to_atomic = get_atomic_num_id("./config/atomic_num_qm9.json")
# elif data_name == "zinc250k":
#     max_atoms = 38
else:
    raise ValueError("[ERROR] Unexpected value data_name={}".format(data_name))

if data_type == "relgraph":
    preprocessor = RelationalGraphPreprocessor(out_size=max_atoms, kekulize=kekulize, id_to_atomic_num=id_to_atomic)
else:
    raise ValueError("[ERROR] Unexpected value data_type={}".format(data_type))

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
download.set_dataset_root(data_dir)
dataset_name = "{}_{}{}.npz".format(data_name, data_type, "_kekulized" if kekulize else "")
os.makedirs(data_dir, exist_ok=True)
print("Save dataset '{}' with format '{}' in '{}' as '{}'".format(data_name, data_type, data_dir, dataset_name))

if data_name == 'qm9':
    dataset = get_qm9(preprocessor, target_index=[1,2,3,4,5])
# elif data_name == 'zinc250k':
#     dataset = datasets.get_zinc250k(preprocessor)
else:
    raise ValueError("[ERROR] Unexpected value data_name={}".format(data_name))

NumpyTupleDataset.save(os.path.join(data_dir, dataset_name), dataset)
