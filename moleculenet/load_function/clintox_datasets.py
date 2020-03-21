"""
Clinical Toxicity (clintox) dataset loader.
@author Caleb Geniesse
"""
import os
import logging
import moleculenet 

logger = logging.getLogger(__name__)

DEFAULT_DIR = moleculenet.utils.get_data_dir()
CLINTOX_URL = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/clintox.csv.gz'


def load_clintox(featurizer='ECFP',
                 split='index',
                 reload=True,
                 data_dir=None,
                 save_dir=None,
                 **kwargs):
  """Load clintox datasets."""
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  if reload:
    save_folder = os.path.join(save_dir, "clintox-featurized", featurizer)
    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

  dataset_file = os.path.join(data_dir, "clintox.csv.gz")
  if not os.path.exists(dataset_file):
    moleculenet.utils.download_url(url=CLINTOX_URL, dest_dir=data_dir)

  logger.info("About to load clintox dataset.")
  dataset = moleculenet.utils.save.load_from_disk(dataset_file)
  clintox_tasks = dataset.columns.values[1:].tolist()
  logger.info("Tasks in dataset: %s" % (clintox_tasks))
  logger.info("Number of tasks in dataset: %s" % str(len(clintox_tasks)))
  logger.info("Number of examples in dataset: %s" % str(dataset.shape[0]))
  if reload:
    loaded, all_dataset, transformers = moleculenet.utils.save.load_dataset_from_disk(
        save_folder)
    if loaded:
      return clintox_tasks, all_dataset, transformers
  # Featurize clintox dataset
  logger.info("About to featurize clintox dataset.")
  if featurizer == 'ECFP':
    featurizer = moleculenet.featurizers.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = moleculenet.featurizers.ConvMolFeaturizer()
  elif featurizer == 'Weave':
    featurizer = moleculenet.featurizers.WeaveFeaturizer()
  elif featurizer == 'Raw':
    featurizer = moleculenet.featurizers.RawFeaturizer()
  elif featurizer == "smiles2img":
    img_spec = kwargs.get("img_spec", "std")
    img_size = kwargs.get("img_size", 80)
    featurizer = moleculenet.featurizers.SmilesToImage(
        img_size=img_size, img_spec=img_spec)

  loader = moleculenet.data.CSVLoader(
      tasks=clintox_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)

  # Transform clintox dataset
  if split is None:
    transformers = [
        moleculenet.transformers.BalancingTransformer(transform_w=True, dataset=dataset)
    ]

    logger.info("Split is None, about to transform data.")
    for transformer in transformers:
      dataset = transformer.transform(dataset)

    return clintox_tasks, (dataset, None, None), transformers

  splitters = {
      'index': moleculenet.splitters.IndexSplitter(),
      'random': moleculenet.splitters.RandomSplitter(),
      'scaffold': moleculenet.splitters.ScaffoldSplitter(),
      'stratified': moleculenet.splitters.RandomStratifiedSplitter()
  }
  splitter = splitters[split]
  logger.info("About to split data with {} splitter.".format(split))
  train, valid, test = splitter.train_valid_test_split(dataset)

  transformers = [
      moleculenet.transformers.BalancingTransformer(transform_w=True, dataset=train)
  ]

  logger.info("About to transform data.")
  for transformer in transformers:
    train = transformer.transform(train)
    valid = transformer.transform(valid)
    test = transformer.transform(test)

  if reload:
    moleculenet.utils.save_dataset_to_disk(save_folder, train, valid, test,
                                             transformers)

  return clintox_tasks, (train, valid, test), transformers
