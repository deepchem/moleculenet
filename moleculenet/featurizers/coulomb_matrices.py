"""
Generate coulomb matrices for molecules.

See Montavon et al., _New Journal of Physics_ __15__ (2013) 095003.
"""
__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "MIT"

import numpy as np
import deepchem as dc
from deepchem.feat import Featurizer
from deepchem.utils import pad_array
from deepchem.feat.atomic_coordinates import AtomicCoordinates


class BPSymmetryFunctionInput(Featurizer):
  """
  Calculate Symmetry Function for each atom in the molecules
  Methods described in https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.98.146401
  """

  def __init__(self, max_atoms):
    self.max_atoms = max_atoms

  def _featurize(self, mol):
    coordfeat = AtomicCoordinates()
    coordinates = coordfeat._featurize(mol)[0]
    atom_numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
    atom_numbers = np.expand_dims(atom_numbers, axis=1)
    assert atom_numbers.shape[0] == coordinates.shape[0]
    n_atoms = atom_numbers.shape[0]
    features = np.concatenate([atom_numbers, coordinates], axis=1)
    return np.pad(features, ((0, self.max_atoms - n_atoms), (0, 0)), 'constant')


class CoulombMatrix(Featurizer):
  """
  Calculate Coulomb matrices for molecules.

  Parameters
  ----------
  max_atoms : int
      Maximum number of atoms for any molecule in the dataset. Used to
      pad the Coulomb matrix.
  remove_hydrogens : bool, optional (default False)
      Whether to remove hydrogens before constructing Coulomb matrix.
  randomize : bool, optional (default False)
      Whether to randomize Coulomb matrices to remove dependence on atom
      index order.
  upper_tri : bool, optional (default False)
      Whether to return the upper triangular portion of the Coulomb matrix.
  n_samples : int, optional (default 1)
      Number of random Coulomb matrices to generate if randomize is True.
  seed : int, optional
      Random seed.

  Example:

  >>> featurizers = dc.feat.CoulombMatrix(max_atoms=23)
  >>> input_file = 'deepchem/feat/tests/data/water.sdf' # really backed by water.sdf.csv
  >>> tasks = ["atomization_energy"]
  >>> featurizer = dc.data.SDFLoader(tasks, smiles_field="smiles", mol_field="mol",
  ...                                    featurizer=featurizers, verbose=False)
  >>> dataset = featurizer.featurize(input_file)
  Reading structures from deepchem/feat/tests/data/water.sdf.
  Featurizing sample 0
  """
  conformers = True
  name = 'coulomb_matrix'

  def __init__(self,
               max_atoms,
               remove_hydrogens=False,
               randomize=False,
               upper_tri=False,
               n_samples=1,
               seed=None):
    self.max_atoms = int(max_atoms)
    self.remove_hydrogens = remove_hydrogens
    self.randomize = randomize
    self.upper_tri = upper_tri
    self.n_samples = n_samples
    if seed is not None:
      seed = int(seed)
    self.seed = seed

  def _featurize(self, mol):
    """
    Calculate Coulomb matrices for molecules. If extra randomized
    matrices are generated, they are treated as if they are features
    for additional conformers.

    Since Coulomb matrices are symmetric, only the (flattened) upper
    triangular portion is returned.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """
    features = self.coulomb_matrix(mol)
    if self.upper_tri:
      features = [f[np.triu_indices_from(f)] for f in features]
    features = np.asarray(features)
    return features

  def coulomb_matrix(self, mol):
    """
    Generate Coulomb matrices for each conformer of the given molecule.

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """
    from rdkit import Chem
    if self.remove_hydrogens:
      mol = Chem.RemoveHs(mol)
    n_atoms = mol.GetNumAtoms()
    z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    rval = []
    for conf in mol.GetConformers():
      d = self.get_interatomic_distances(conf)
      m = np.zeros((n_atoms, n_atoms))
      for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
          if i == j:
            m[i, j] = 0.5 * z[i]**2.4
          elif i < j:
            m[i, j] = (z[i] * z[j]) / d[i, j]
            m[j, i] = m[i, j]
          else:
            continue
      if self.randomize:
        for random_m in self.randomize_coulomb_matrix(m):
          random_m = pad_array(random_m, self.max_atoms)
          rval.append(random_m)
      else:
        m = pad_array(m, self.max_atoms)
        rval.append(m)
    rval = np.asarray(rval)
    return rval

  def randomize_coulomb_matrix(self, m):
    """
    Randomize a Coulomb matrix as decribed in Montavon et al., _New Journal
    of Physics_ __15__ (2013) 095003:

        1. Compute row norms for M in a vector row_norms.
        2. Sample a zero-mean unit-variance noise vector e with dimension
           equal to row_norms.
        3. Permute the rows and columns of M with the permutation that
           sorts row_norms + e.

    Parameters
    ----------
    m : ndarray
        Coulomb matrix.
    n_samples : int, optional (default 1)
        Number of random matrices to generate.
    seed : int, optional
        Random seed.
    """
    rval = []
    row_norms = np.asarray([np.linalg.norm(row) for row in m], dtype=float)
    rng = np.random.RandomState(self.seed)
    for i in range(self.n_samples):
      e = rng.normal(size=row_norms.size)
      p = np.argsort(row_norms + e)
      new = m[p][:, p]  # permute rows first, then columns
      rval.append(new)
    return rval

  @staticmethod
  def get_interatomic_distances(conf):
    """
    Get interatomic distances for atoms in a molecular conformer.

    Parameters
    ----------
    conf : RDKit Conformer
        Molecule conformer.
    """
    n_atoms = conf.GetNumAtoms()
    coords = [
        conf.GetAtomPosition(i).__idiv__(0.52917721092) for i in range(n_atoms)
    ]  # Convert AtomPositions from Angstrom to bohr (atomic units)
    d = np.zeros((n_atoms, n_atoms), dtype=float)
    for i in range(n_atoms):
      for j in range(n_atoms):
        if i < j:
          d[i, j] = coords[i].Distance(coords[j])
          d[j, i] = d[i, j]
        else:
          continue
    return d


class CoulombMatrixEig(CoulombMatrix):
  """
  Calculate the eigenvales of Coulomb matrices for molecules.

  Parameters
  ----------
  max_atoms : int
      Maximum number of atoms for any molecule in the dataset. Used to
      pad the Coulomb matrix.
  remove_hydrogens : bool, optional (default False)
      Whether to remove hydrogens before constructing Coulomb matrix.
  randomize : bool, optional (default False)
      Whether to randomize Coulomb matrices to remove dependence on atom
      index order.
  n_samples : int, optional (default 1)
      Number of random Coulomb matrices to generate if randomize is True.
  seed : int, optional
      Random seed.

  Example:

  >>> featurizers = dc.feat.CoulombMatrixEig(max_atoms=23)
  >>> input_file = 'deepchem/feat/tests/data/water.sdf' # really backed by water.sdf.csv
  >>> tasks = ["atomization_energy"]
  >>> featurizer = dc.data.SDFLoader(tasks, smiles_field="smiles", mol_field="mol",
  ...                                    featurizer=featurizers, verbose=False)
  >>> dataset = featurizer.featurize(input_file)
  Reading structures from deepchem/feat/tests/data/water.sdf.
  Featurizing sample 0

  """

  conformers = True
  name = 'coulomb_matrix'

  def __init__(self,
               max_atoms,
               remove_hydrogens=False,
               randomize=False,
               n_samples=1,
               seed=None):
    self.max_atoms = int(max_atoms)
    self.remove_hydrogens = remove_hydrogens
    self.randomize = randomize
    self.n_samples = n_samples
    if seed is not None:
      seed = int(seed)
    self.seed = seed

  def _featurize(self, mol):
    """
    Calculate eigenvalues of Coulomb matrix for molecules. Eigenvalues
    are returned sorted by absolute value in descending order and padded
    by max_atoms. 

    Parameters
    ----------
    mol : RDKit Mol
        Molecule.
    """
    cmat = self.coulomb_matrix(mol)
    features = []
    for f in cmat:
      w, v = np.linalg.eig(f)
      w_abs = np.abs(w)
      sortidx = np.argsort(w_abs)
      sortidx = sortidx[::-1]
      w = w[sortidx]
      f = pad_array(w, self.max_atoms)
      features.append(f)
    features = np.asarray(features)
    return features
