from moleculenet.featurizers import Featurizer


class RawFeaturizer(Featurizer):

  def __init__(self, smiles=False):
    self.smiles = smiles

  def _featurize(self, mol):
    from rdkit import Chem
    if self.smiles:
      return Chem.MolToSmiles(mol)
    else:
      return mol
