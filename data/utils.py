import json
import logging as log
import os

import chainer
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import Atom, RWMol
from chainer_chemistry.datasets import NumpyTupleDataset

import data.relational_graph_preprocessor as molecule_preprocessor

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
        sigma_diag = xp.sqrt(xp.exp(model.ln_var.data))

    sigma = temp * sigma_diag

    with chainer.no_backprop_mode():
        if z_mu is not None:
            mu = z_mu
            sigma = 0.01 * xp.eye(z_dim, dtype=xp.float32)
        z = xp.random.normal(mu, sigma, (batch_size, z_dim)).astype(xp.float32)
        x, adj = model.reverse(z, true_adj=true_adj)
    return x, adj


def construct_mol(atom_id: np.ndarray, A, atomic_num_list):
    mol = RWMol()
    atomic_num_list = np.array(atomic_num_list, dtype=np.int)
    if atom_id.dtype != np.int:
        atom_id = atom_id.astype(np.int)
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
        if i > j and adj[i, j] > 0:
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
    valid = [Chem.MolToSmiles(construct_mol(x_elem.astype(np.int), adj_elem, atomic_num_list), isomericSmiles=False, canonical=True)
             for x_elem, adj_elem in data]
    return valid


def smiles_to_adj(smiles, max_atoms, out_size, atom_id_to_atomic_num_list):
    preprocessor = molecule_preprocessor.RelationalGraphPreprocessor(
        max_atoms=max_atoms, out_size=out_size, 
        id_to_atomic_num=atom_id_to_atomic_num_list, kekulize=True)
    
    return preprocessor.get_input_features(Chem.MolFromSmiles(smiles))


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


def get_latent_vec(model, smiles, max_atoms, out_size, atom_id_to_atomic_num):
    x, adj = smiles_to_adj(smiles, max_atoms, out_size, atom_id_to_atomic_num)
    x = np.expand_dims(x, axis=0)
    adj = np.expand_dims(adj, axis=0)
    with chainer.no_backprop_mode():
        z = model(x, adj)
    return np.hstack([z[0][0].data, z[0][1].data]).squeeze(0)


def gen_direction(latent_size):
    """ get two orthogonal direction """
    x = np.random.randn(latent_size).astype(np.float32)
    x /= np.linalg.norm(x)

    y = np.random.randn(latent_size).astype(np.float32)
    y -= y.dot(x) * x
    y /= np.linalg.norm(y)
    return x, y


def gen_neiborhood(v, x, y, step, r):
    step_vec = np.expand_dims(np.arange(-r, r+1, 1, dtype=np.float32) * step, 1) # (2r+1, 1)
    diff_x = np.matmul(step_vec, np.expand_dims(x, 0)) # (2r+1, w)
    diff_y = np.matmul(step_vec, np.expand_dims(y, 0)) # (2r+1, w)
    diff_tensor = np.repeat(diff_x, 2*r+1, 0).reshape(2*r+1, 2*r+1, -1) + np.transpose(np.repeat(diff_y, 2*r+1, 0).reshape(2*r+1, 2*r+1, -1), (1,0,2)) # (2r+1, 2r+1, w)
    return v + diff_tensor


def generate_mols_interpolation(model, z0=None, true_adj=None, device=-1, seed=0, mols_per_row=13, delta=1.):
    np.random.seed(seed)
    adj_size, x_size = model.adj_size, model.x_size
    latent_size = adj_size + x_size
    if z0 is None:
        zx_mu = np.zeros([x_size], dtype=np.float32)
        zx_sigma = float(model.x_var) * np.eye(x_size, dtype=np.float32)
        zx = np.random.multivariate_normal(zx_mu, zx_sigma).astype(np.float32)
    else:
        zx, zadj_true = np.split(z0, [x_size])

    x, y = gen_direction(latent_size)
    zx_x, zadj_x = np.split(x, [x_size])
    zx_y, zadj_y = np.split(y, [x_size])
    interpolations_x = gen_neiborhood(zx, zx_x, zx_y, delta, mols_per_row // 2).reshape(-1, x_size)
    adj_steps = gen_neiborhood(np.zeros([adj_size], dtype=np.float32), zadj_x, zadj_y, delta, mols_per_row // 2).reshape(-1, adj_size)

    if device >= 0:
        interpolations_x = chainer.backends.cuda.to_gpu(interpolations_x, device)
        adj_steps = chainer.backends.cuda.to_gpu(adj_steps, device)
        if not z0 is None:
            zadj_true = chainer.backends.cuda.to_gpu(zadj_true, device)

    interpolations_adj = model.latent_trans(interpolations_x).array
    xp = chainer.backends.cuda.get_array_module(interpolations_adj)
    if z0 is None:
        zadj_sigma = model.adj_var * xp.eye(adj_size, dtype=np.float32)
        interpolations_adj += xp.random.multivariate_normal(xp.zeros([adj_size]), zadj_sigma).astype(xp.float32)
    else:
        interpolations_adj = interpolations_adj - interpolations_adj[interpolations_adj.shape[0] // 2] + zadj_true
    interpolations_adj += adj_steps
    # assert xp.prod(xp.equal(interpolations_adj[interpolations_adj.shape[0] // 2], zadj_true))
    interpolations = xp.concatenate([interpolations_x, interpolations_adj], axis=-1)
    x, adj = model.reverse(interpolations, true_adj=true_adj, norm_sample=False)
    return x, adj


def visualize_interpolation(filepath: str, model: chainer.Chain, atomic_num_id: list, max_atom: int, out_size: int, mol_smiles=None, mols_per_row=13,
                            delta=0.1, seed=0, mol_dataset=None, save_mol_fig=True, device=-1):
    z0 = None
    if mol_smiles is not None:
        z0 = get_latent_vec(model, mol_smiles, max_atom, out_size, atomic_num_id)
    else:
        with chainer.no_backprop_mode(), chainer.using_config("train", False):
            np.random.seed(seed)
            mol_index = np.random.randint(0, len(mol_dataset))
            x = np.expand_dims(mol_dataset[mol_index][0], axis=0)
            adj = np.expand_dims(mol_dataset[mol_index][1], axis=0)
            if device >= 0:
                adj = chainer.backends.cuda.to_gpu(adj, device)
                x = chainer.backends.cuda.to_gpu(x, device)
            z0 = model(x, adj)[0]
            if device >= 0:
                z0 = chainer.backends.cuda.cupy.hstack((z0[0].data, z0[1].data)).squeeze(0)
                z0 = chainer.backends.cuda.to_cpu(z0).astype(np.float32)
            else:
                z0 = np.hstack((z0[0].data, z0[1].data)).squeeze(0)
        
    x, adj = generate_mols_interpolation(model, z0=z0, mols_per_row=mols_per_row, delta=delta, seed=seed, device=device)
    adj = _to_numpy_array(adj, device=device)
    x = _to_numpy_array(x, device=device)
    interpolation_mols = [valid_mol(construct_mol(x_elem, adj_elem, atomic_num_id)) 
                          for x_elem, adj_elem in zip(x, adj)]
    valid_mols = [mol for mol in interpolation_mols if mol is not None]
    print("interpolation_mols valid {}/{}".format(len(valid_mols), len(interpolation_mols)))
    img = Draw.MolsToGridImage(interpolation_mols, molsPerRow=mols_per_row, subImgSize=(250, 250))
    if save_mol_fig:
        img.save(filepath)
    else:
        img.show()


def _to_numpy_array(x, device=-1):
    if isinstance(x, chainer.Variable):
        x = x.array
    if device >= 0:
        return chainer.backends.cuda.to_cpu(x)
    return x


def save_mol_png(mol, filepath, size=(600, 600)):
    Draw.MolToFile(mol, filepath, size=size)


def load_dataset(dataset_path, validation_idxs):
    dataset = NumpyTupleDataset.load(dataset_path)
    if validation_idxs:
        train_idxs = [i for i in range(
            len(dataset)) if i not in validation_idxs]
        trainset_size = len(train_idxs)
        train_idxs.extend(validation_idxs)
        trainset, valset = chainer.datasets.split_dataset(
            dataset, trainset_size, train_idxs)
    else:
        trainset, valset = chainer.datasets.split_dataset_random(
            dataset, int(len(dataset) * 0.8), seed=777)
    return trainset, valset, dataset


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
