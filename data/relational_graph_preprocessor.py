import numpy as np
from chainer_chemistry.dataset.preprocessors.common import construct_atomic_number_array
from chainer_chemistry.dataset.preprocessors.common import MolFeatureExtractionError  # NOQA
from chainer_chemistry.dataset.preprocessors.common import type_check_num_atoms
from chainer_chemistry.dataset.preprocessors.mol_preprocessor import MolPreprocessor
from rdkit.Chem.rdmolops import GetAdjacencyMatrix


def get_adj_info(mol):
    """Return information about adjacency matrix of mol
    Args:
        mol (Chem.Mol)
    Returns (map)
    """
    bonds = mol.GetBonds()
    # (start_node_idx, end_node_idx, bond_type)
    # bond_type: 1.0 for SINGLE, 1.5 for AROMATIC, 2.0 for DOUBLE, 3.0 for TRIPLE
    adj_info = map(lambda b:(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble()), bonds)
    return adj_info

def construct_atom_id_array_func(id_to_atomic_num: list):
    def construct_atom_id_array(mol, out_size=-1):
        atomic_num_array = construct_atomic_number_array(mol, out_size)
        atom_id_array = np.zeros(atomic_num_array.shape, dtype=np.float32)
        for i in range(atomic_num_array.shape[0]):
            atom_id_array[i] = id_to_atomic_num.index(atomic_num_array[i])
        return atom_id_array
    return construct_atom_id_array

def construct_discrete_edge_matrix(mol, out_size=-1, kekulize=False):
    """construct adjacency tensor. In addition to NxN adjacency matrix,
    third dimension stores one-hot encoded bond type.
    - channel 0 for No link (virtual link)
    - channel 1 for SINGLE
    - channel 2 for DOUBLE
    - channel 3 for TRIPLE
    - channel 4 for AROMATIC, if not kekulize
    The output adjacency matrix has no self-connection.
    Args:
        mol (Chem.Mol):
        out_size (int):
    Returns (numpy.ndarray):
    """
    if mol is None:
        raise MolFeatureExtractionError("mol is None")
    N = mol.GetNumAtoms()

    if kekulize:
        num_edge_type = 4
    else:
        num_edge_type = 5

    if out_size < 0:
        size = N
    elif out_size >= N:
        size = out_size
    else:
        raise MolFeatureExtractionError('out_size {} is smaller than number '
                                        'of atoms in mol {}'
                                        .format(out_size, N))

    adjs = np.zeros((num_edge_type, size, size), dtype=np.float32)
    adjs[0] += 1 # virtual link
    adjs[0] -= np.eye(size) # remove self-connection from virtual link 
    for info in get_adj_info(mol):
        s, e, b = info
        if b == 1.5: # AROMATIC
            b = num_edge_type-1
        adjs[int(b), s, e] = 1.0
        adjs[int(b), e, s] = 1.0
        adjs[0, s, e] = 0.0
        adjs[0, e, s] = 0.0
    
    return adjs


class RelationalGraphPreprocessor(MolPreprocessor):
    """Preprocessor for relational graph (A_{n x n x R}, X_{n x F})
    Args:
        max_atoms (int): Max number of atoms for each molecule, if the
            number of atoms is more than this value, this data is simply
            ignored.
            Setting negative value indicates no limit for max atoms.
        out_size (int): It specifies the size of array returned by
            `get_input_features`.
            If the number of atoms in the molecule is less than this value,
            the returned arrays is padded to have fixed size.
            Setting negative value indicates do not pad returned array.
        add_Hs (bool): If True, implicit Hs are added.
        kekulize (bool): If True, Kekulizes the molecule.
    """

    def __init__(self, max_atoms=-1, out_size=-1, add_Hs=False, id_to_atomic_num=None, kekulize=False):
        super(RelationalGraphPreprocessor, self).__init__(add_Hs=add_Hs, kekulize=kekulize)
        if max_atoms >= 0 and 0 <= out_size < max_atoms:
            raise ValueError('max_atoms {} must be less or equal to '
                             'out_size {}'.format(max_atoms, out_size))
        self.max_atoms = max_atoms
        self.out_size = out_size
        self.kekulize = kekulize
        self.id_to_atomic_num = id_to_atomic_num
        if id_to_atomic_num is None:
            self.atom_feature_func = construct_atomic_number_array
        else:
            self.atom_feature_func = construct_atom_id_array_func(id_to_atomic_num)

    def get_input_features(self, mol):
        """get input features
        Args:
            mol (rdkit.Chem.rdchem.Mol)
        Returns:
            input features
        """
        type_check_num_atoms(mol, self.max_atoms)
        atom_array = self.atom_feature_func(mol, out_size=self.out_size)
        adj_tensor = construct_discrete_edge_matrix(mol, out_size=self.out_size, kekulize=self.kekulize)
        return (atom_array, adj_tensor)

if __name__ == "__main__":
    import rdkit
    import rdkit.Chem as Chem
    # m = Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    m = Chem.MolFromSmiles("C#C")
    preprocessor = RelationalGraphPreprocessor(add_Hs=True, kekulize=True)
    norm_smiles, mol = preprocessor.prepare_smiles_and_mol(m)
    atom_features, adjs = preprocessor.get_input_features(mol)
    print(atom_features)
    print(adjs)


    import chainer
    import chainer.functions as F
    from chainer import cuda
    from chainer_chemistry import MAX_ATOMIC_NUM
    from chainer_chemistry.links import EmbedAtomID
    from chainer_chemistry.links import GraphLinear

    def rescale_adj(adj):
        xp = cuda.get_array_module(adj)
        num_neighbors = F.sum(adj, axis=(1, 2))
        batch_size, num_edge_type, num_node, _ = adj.shape
        base = xp.ones(num_neighbors.shape, dtype=xp.float32)
        cond = num_neighbors.array != 0
        num_neighbors_inv = F.reshape(1 / F.where(cond, num_neighbors, base), (batch_size, 1, 1, num_node))
        return adj * F.broadcast_to(num_neighbors_inv, adj.shape)

    print(rescale_adj(np.reshape(adjs, [1] + list(adjs.shape))))