"""
Thermodynamic Solubility Dataset Loader
"""
import os
import logging
import deepchem
import numpy as np

logger = logging.getLogger(__name__)

THERMOSOL_URL = "http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/thermosol.csv"
DEFAULT_DATA_DIR = deepchem.utils.get_data_dir()


def remove_missing_entries(dataset):
  """Remove missing entries.

  Some of the datasets have missing entries that sneak in as zero'd out
  feature vectors. Get rid of them.
  """
  for i, (X, y, w, ids) in enumerate(dataset.itershards()):
    available_rows = X.any(axis=1)
    logger.info("Shard %d has %d missing entries." %
                (i, np.count_nonzero(~available_rows)))
    X = X[available_rows]
    y = y[available_rows]
    w = w[available_rows]
    ids = ids[available_rows]
    dataset.set_shard(i, X, y, w, ids)


def load_thermosol(featurizer="ECFP",
                   data_dir=None,
                   save_dir=None,
                   split=None,
                   split_seed=None,
                   reload=True):
  """Loads the thermodynamic solubility datasets."""
  # Featurizer thermosol dataset
  logger.info("About to featurize thermosol dataset...")
  thermosol_tasks = ["target"]  #Task is solubility in pH 7.4 buffer

  if data_dir is None:
    data_dir = DEFAULT_DATA_DIR
  if save_dir is None:
    save_dir = DEFAULT_DATA_DIR

  if reload:
    save_folder = os.path.join(save_dir, "thermosol-featurized", featurizer)
    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return thermosol_tasks, all_dataset, transformers

  dataset_file = os.path.join(data_dir, "thermosol.csv")
  if not os.path.exists(dataset_file):
    logger.info("{} does not exist. Downloading it.".format(dataset_file))
    deepchem.utils.download_url(url=THERMOSOL_URL, dest_dir=data_dir)

  if featurizer == 'ECFP':
    featurizer = deepchem.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = deepchem.feat.ConvMolFeaturizer()
  elif featurizer == 'Weave':
    featurizer = deepchem.feat.WeaveFeaturizer()
  elif featurizer == 'Raw':
    featurizer = deepchem.feat.RawFeaturizer()
  elif featurizer == 'AdjacencyConv':
    featurizer = deepchem.feat.AdjacencyFingerprint(
        max_n_atoms=150, max_valence=6)
  elif featurizer == "smiles2img":
    img_spec = kwargs.get("img_spec", "std")
    img_size = kwargs.get("img_size", 80)
    featurizer = deepchem.feat.SmilesToImage(
        img_size=img_size, img_spec=img_spec)

  logger.info("Featurizing datasets.")
  loader = deepchem.data.CSVLoader(
      tasks=thermosol_tasks, smiles_field='smile', featurizer=featurizer)
  dataset = loader.featurize(input_files=[dataset_file], shard_size=2000)

  logger.info("Removing missing entries...")
  remove_missing_entries(dataset)

  if split == None:
    logger.info("About to transform the data...")
    transformers = []
    for transformer in transformers:
      logger.info("Transforming the dataset with transformer ",
                  transformer.__class__.__name__)
      dataset = transformer.transform(dataset)
    return thermosol_tasks, (dataset, None, None), transformers

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'scaffold': deepchem.splits.ScaffoldSplitter(),
      'butina': deepchem.splits.ButinaSplitter(),
      'stratified': deepchem.splits.SingletaskStratifiedSplitter()
  }
  splitter = splitters[split]
  frac_train = kwargs.get("frac_train", 0.8)
  frac_valid = kwargs.get('frac_valid', 0.1)
  frac_test = kwargs.get('frac_test', 0.1)

  train, valid, test = splitter.train_valid_test_split(
      dataset,
      frac_train=frac_train,
      frac_valid=frac_valid,
      frac_test=frac_test,
      seed=split_seed)
  transformers = []

  logger.info("About to transform the data...")
  for transformer in transformers:
    logger.info("Transforming the data with transformer ",
                transformer.__class__.__name__)
    train = transformer.transform(train)
    valid = transformer.transform(valid)
    test = transformer.transform(test)

  if reload:
    logger.info("Saving file to {}.".format(save_folder))
    deepchem.utils.save.save_dataset_to_disk(save_folder, train, valid, test,
                                             transformers)
  return thermosol_tasks, (train, valid, test), transformers
