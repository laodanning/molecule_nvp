import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import chainer
import numpy as np
from model.atom_embed.atom_embed import atom_embed_model, AtomEmbedModel
from model.hyperparameter import Hyperparameter
# from data.utils import get_atomic_num_id

default_model_path = "./output/qm9/final_embed_model.npz"
default_hyper_params_path = "./output/qm9/atom_embed_model_hyper.json"
default_atomic_num_list_path = "./config/atomic_num_qm9.json"

def gen_direction(latent_size):
    """ get two orthogonal direction """
    x = np.random.randn(latent_size)
    x /= np.linalg.norm(x)

    y = np.random.randn(latent_size)
    y -= y.dot(x) * x
    y /= np.linalg.norm(y)
    return x, y

def gen_neiborhood(v, x, y, step, r):
    step_vec = np.expand_dims(np.arange(-r, r+1, 1) * step, 1) # (2r+1, 1)
    diff_x = np.matmul(step_vec, np.expand_dims(x, 0)) # (2r+1, w)
    diff_y = np.matmul(step_vec, np.expand_dims(y, 0)) # (2r+1, w)
    diff_tensor = np.repeat(diff_x, 2*r+1, 0).reshape(2*r+1, 2*r+1, -1) + np.transpose(np.repeat(diff_y, 2*r+1, 0).reshape(2*r+1, 2*r+1, -1), (1,0,2)) # (2r+1, 2r+1, w)
    return v + diff_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, default=default_model_path, help="path to embedding model")
    parser.add_argument("-p", "--params_path", type=str, default=default_hyper_params_path, help="path to hyper parameters")
    parser.add_argument("-a", "--atomic_num_list", type=str, default=default_atomic_num_list_path, help="path to atomic num list")
    args = parser.parse_args()

    hyper_params = Hyperparameter(args.params_path)
    model = atom_embed_model(hyper_params)
    chainer.serializers.load_npz(args.model_path, model)
    # atomic_num_list = get_atomic_num_id(args.atomic_num_list)

    # print(model.embed.embed.W)
    words = model.embed.embed.W.array
    num_atom_types, word_size = words.shape
    norm = np.linalg.norm(words, axis=1, keepdims=True)
    norm_matrix = np.matmul(norm, norm.T)

    # cor_matrix = np.matmul(words, words.T) / norm_matrix
    # print(cor_matrix)

    x, y = gen_direction(word_size)
    print(x.dot(y))

    step = 1e-2
    r = 2
    print(words[3])
    neighbors = gen_neiborhood(words[3], x, y, step ,r)
    print(neighbors[2,2])
    print(neighbors.shape)
