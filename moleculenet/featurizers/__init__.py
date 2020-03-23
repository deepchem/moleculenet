"""
Making it easy to import in classes.
"""

from moleculenet.featurizers.base_classes import Featurizer
from moleculenet.featurizers.base_classes import ComplexFeaturizer
from moleculenet.featurizers.base_classes import UserDefinedFeaturizer
from moleculenet.featurizers.basic import RDKitDescriptors
from moleculenet.featurizers.one_hot import OneHotFeaturizer
from moleculenet.featurizers.graph_features import ConvMolFeaturizer
from moleculenet.featurizers.graph_features import WeaveFeaturizer
from moleculenet.featurizers.smiles_featurizers import SmilesToSeq
from moleculenet.featurizers.smiles_featurizers import SmilesToImage
