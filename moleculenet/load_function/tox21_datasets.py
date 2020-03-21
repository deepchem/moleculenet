"""
Tox21 dataset loader.
"""
import os
import logging
import moleculenet 

logger = logging.getLogger(__name__)

TOX21_URL = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/tox21.csv.gz'
DEFAULT_DIR = moleculenet.utils.get_data_dir()


def load_tox21(featurizer='ECFP',
               split='index',
               reload=True,
               K=4,
               data_dir=None,
               save_dir=None,
               **kwargs):
  """Load Tox21 datasets. Does not do train/test split"""
  # Featurize Tox21 dataset

  tox21_tasks = [
      'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
      'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
  ]

  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  if reload:
    save_folder = os.path.join(save_dir, "tox21-featurized", str(featurizer))
    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

    loaded, all_dataset, transformers = moleculenet.utils.load_dataset_from_disk(
        save_folder)
    if loaded:
      return tox21_tasks, all_dataset, transformers

  dataset_file = os.path.join(data_dir, "tox21.csv.gz")
  if not os.path.exists(dataset_file):
    moleculenet.utils.download_url(url=TOX21_URL, dest_dir=data_dir)

  if featurizer == 'ECFP':
    featurizer = moleculenet.featurizers.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = moleculenet.featurizers.ConvMolFeaturizer()
  elif featurizer == 'Weave':
    featurizer = moleculenet.featurizers.WeaveFeaturizer()
  elif featurizer == 'Raw':
    featurizer = moleculenet.featurizers.RawFeaturizer()
  elif featurizer == 'AdjacencyConv':
    featurizer = moleculenet.featurizers.AdjacencyFingerprint(
        max_n_atoms=150, max_valence=6)
  elif featurizer == "smiles2img":
    img_size = kwargs.get("img_size", 80)
    img_spec = kwargs.get("img_spec", "std")
    featurizer = moleculenet.featurizers.SmilesToImage(
        img_size=img_size, img_spec=img_spec)

  loader = moleculenet.data.CSVLoader(
      tasks=tox21_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)

  if split == None:
    # Initialize transformers
    transformers = [
        moleculenet.trans.BalancingTransformer(transform_w=True, dataset=dataset)
    ]

    logger.info("About to transform data")
    for transformer in transformers:
      dataset = transformer.transform(dataset)

    return tox21_tasks, (dataset, None, None), transformers

  splitters = {
      'index': moleculenet.splitters.IndexSplitter(),
      'random': moleculenet.splitters.RandomSplitter(),
      'scaffold': moleculenet.splitters.ScaffoldSplitter(),
      'butina': moleculenet.splitters.ButinaSplitter(),
      'task': moleculenet.splitters.TaskSplitter(),
      'stratified': moleculenet.splitters.RandomStratifiedSplitter()
  }
  splitter = splitters[split]
  if split == 'task':
    fold_datasets = splitter.k_fold_split(dataset, K)
    all_dataset = fold_datasets
  else:
    frac_train = kwargs.get("frac_train", 0.8)
    frac_valid = kwargs.get('frac_valid', 0.1)
    frac_test = kwargs.get('frac_test', 0.1)

    train, valid, test = splitter.train_valid_test_split(
        dataset,
        frac_train=frac_train,
        frac_valid=frac_valid,
        frac_test=frac_test)
    all_dataset = (train, valid, test)

    transformers = [
        moleculenet.transformers.BalancingTransformer(transform_w=True, dataset=train)
    ]

    logger.info("About to transform data")
    for transformer in transformers:
      train = transformer.transform(train)
      valid = transformer.transform(valid)
      test = transformer.transform(test)

    if reload:
      moleculenet.utils.save_dataset_to_disk(save_folder, train, valid, test,
                                               transformers)
  return tox21_tasks, all_dataset, transformers
