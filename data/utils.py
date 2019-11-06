import json
import logging as log
import os

import chainer
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import Atom, RWMol

""" - channel 0 for No link (virtual link)
    - channel 1 for SINGLE
    - channel 2 for DOUBLE
    - channel 3 for TRIPLE
    - channel 4 for AROMATIC, if not kekulize
"""
adj_to_bond_type = (
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC)


def get_validation_idxs(file_path):
    log.info("loading train/validation split information from {}".format(file_path))
    if os.path.exists(file_path):
        with open(file_path) as f:
            data = json.load(f)
    else:
        log.warning(
            "Cannot find validation index file in {}, divide dataset randomly.".format(file_path))
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
    log.info("loading information between atomic num and atom id "
             "from {}".format(file_path))
    if os.path.exists(file_path):
        with open(file_path) as f:
            data = json.load(f)
    else:
        raise ValueError(
            "Cannot find atomid to atomic number information file in {}".format(file_path))
    return data


def generate_mols(model: chainer.Chain, temp=0.5, z_mu=None, batch_size=20, true_adj=None, device=-1):
    """

    :param model: GraphNVP model
    :param z_mu: latent vector of a molecule
    :param batch_size:
    :param true_adj:
    :param gpu:
    :return:
    """
    device_obj = chainer.backend.get_device(device)
    xp = device_obj.xp
    z_dim = model.adj_size + model.x_size
    mu = xp.zeros([z_dim], dtype=xp.float32)
    sigma_diag = xp.ones([z_dim])

    if model.hyperparams.learn_dist:
        sigma_diag = xp.sqrt(xp.exp(model.ln_var.data)) * sigma_diag
        # sigma_diag = xp.exp(xp.hstack((model.ln_var_x.data, model.ln_var_adj.data)))

    sigma = temp * sigma_diag

    with chainer.no_backprop_mode():
        if z_mu is not None:
            mu = z_mu
            sigma = 0.01 * xp.eye(z_dim, dtype=xp.float32)
        z = xp.random.normal(mu, sigma, (batch_size, z_dim)).astype(xp.float32)
        x, adj = model.reverse(z, true_adj=true_adj)
    return x, adj


def construct_mol(atom_id, A, atomic_num_list):
    mol = RWMol()
    atomic_num_list = np.array(atomic_num_list, dtype=np.int)
    atomic_num = list(filter(lambda x: x > 0, atomic_num_list[atom_id]))
    atoms_exist = np.array(
        list(map(lambda x: x > 0, atomic_num_list[atom_id]))).astype(np.bool)
    atoms_exist = np.expand_dims(atoms_exist, axis=1)
    exist_matrix = atoms_exist * atoms_exist.T

    for atom in atomic_num:
        mol.AddAtom(Atom(int(atom)))

    # A (edge_type, num_node, num_node)
    adj = np.argmax(A, axis=0)
    adj = adj[exist_matrix].reshape(len(atomic_num), len(atomic_num))
    for i, j in zip(*np.nonzero(adj)):
        if i > j:
            mol.AddBond(int(i), int(j), adj_to_bond_type[adj[i, j]])
    return mol


def valid_mol(x):
    s = Chem.MolFromSmiles(Chem.MolToSmiles(x)) if x is not None else None
    if s is not None and '.' not in Chem.MolToSmiles(s):
        return s
    return None


@chainer.dataset.converter()
def molecule_id_converter(batch, device):
    """Convert float atom feature(atom id or atomic number) to int
    """
    batch = chainer.dataset.concat_examples(batch, device)
    return (chainer.functions.cast(batch[0], "int"), batch[1])


def atom_shuffle(data):
    # node: N x F
    # adj: R x N x N
    node, adj = data
    num_atoms = node.shape[0]
    permutation = np.random.permutation(num_atoms)
    return node[permutation], adj[:, permutation]


def adj_to_smiles(data: list, atomic_num_list: list) -> list:
    valid = [Chem.MolToSmiles(construct_mol(x_elem, adj_elem, atomic_num_list))
             for x_elem, adj_elem in data]
    return valid


def check_validity(x, adj, atomic_num_list, device=-1, return_unique=True):
    adj = _to_numpy_array(adj, device)
    x = _to_numpy_array(x, device)
    valid = [valid_mol(construct_mol(x_elem, adj_elem, atomic_num_list))
             for x_elem, adj_elem in zip(x, adj)]
    valid = [mol for mol in valid if mol is not None]
    log.info("valid molecules: {}/{}".format(len(valid), adj.shape[0]))
    for i, mol in enumerate(valid):
        log.debug("[{}] {}".format(i, Chem.MolToSmiles(mol)))

    n_mols = x.shape[0]
    valid_ratio = len(valid)/n_mols
    valid_smiles = [Chem.MolToSmiles(mol) for mol in valid]
    unique_smiles = list(set(valid_smiles))
    unique_ratio = 0.
    if len(valid) > 0:
        unique_ratio = len(unique_smiles)/len(valid)
    if return_unique:
        valid_smiles = unique_smiles
    valid_mols = [Chem.MolFromSmiles(s) for s in valid_smiles]
    log.info("valid: {:.3f}%, unique: {:.3f}%".format(
        valid_ratio * 100, unique_ratio * 100))

    results = dict()
    results['valid_mols'] = valid_mols
    results['valid_smiles'] = valid_smiles
    results['valid_ratio'] = valid_ratio*100
    results['unique_ratio'] = unique_ratio*100

    return results


def check_novelty(gen_smiles, train_smiles):
    if len(gen_smiles) == 0:
        novel_ratio = 0.
    else:
        duplicates = [1 for mol in gen_smiles if mol in train_smiles]
        novel = len(gen_smiles) - sum(duplicates)
        novel_ratio = novel*100./len(gen_smiles)
    log.info("novelty: {}%".format(novel_ratio))
    return novel_ratio


def visualize_interpolation(filepath: str, model: chainer.Chain, atomic_num_id: list, mol_smiles=None, mols_per_row=13,
                            delta=0.1, seed=0, true_data=None, device=-1):
    pass


def _to_numpy_array(x, device=-1):
    if isinstance(x, chainer.Variable):
        x = x.array
    if device >= 0:
        return chainer.backends.cuda.to_cpu(x)
    return x


def save_mol_png(mol, filepath, size=(600, 600)):
    Draw.MolToFile(mol, filepath, size=size)


if __name__ == "__main__":
    N = 3
    E = 4
    C = 5
    x = np.random.randint(0, C, size=N)
    atomic_num_id = [6, 7, 8, 9, 0]
    adj = np.random.randn(E, N, N)
    max_edge = np.max(adj, axis=0)
    adj = (adj == max_edge).astype(np.int)
    mol = construct_mol(x, adj, atomic_num_id)
    save_mol_png(mol, "./{}.png".format(Chem.MolToSmiles(mol)))
