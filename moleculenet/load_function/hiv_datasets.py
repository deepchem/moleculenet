"""
hiv dataset loader.
"""
import os
import logging
import moleculenet 

logger = logging.getLogger(__name__)

HIV_URL = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/HIV.csv'
DEFAULT_DIR = moleculenet.utils.get_data_dir()


def load_hiv(featurizer='ECFP',
             split='index',
             reload=True,
             data_dir=None,
             save_dir=None,
             **kwargs):
  """Load hiv datasets. Does not do train/test split"""
  # Featurize hiv dataset
  logger.info("About to featurize hiv dataset.")
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  hiv_tasks = ["HIV_active"]

  if reload:
    save_folder = os.path.join(save_dir, "hiv-featurized", str(featurizer))
    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

  if reload:
    loaded, all_dataset, transformers = moleculenet.utils.load_dataset_from_disk(
        save_folder)
    if loaded:
      return hiv_tasks, all_dataset, transformers

  dataset_file = os.path.join(data_dir, "HIV.csv")
  if not os.path.exists(dataset_file):
    moleculenet.utils.download_url(url=HIV_URL, dest_dir=data_dir)

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
      tasks=hiv_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)

  if split is None:
    transformers = [
        moleculenet.transformers.BalancingTransformer(transform_w=True, dataset=dataset)
    ]

    logger.info("Split is None, about to transform data")
    for transformer in transformers:
      dataset = transformer.transform(dataset)

    return hiv_tasks, (dataset, None, None), transformers

  splitters = {
      'index': moleculenet.splitters.IndexSplitter(),
      'random': moleculenet.splitters.RandomSplitter(),
      'scaffold': moleculenet.splitters.ScaffoldSplitter(),
      'butina': moleculenet.splitters.ButinaSplitter(),
      'stratified': moleculenet.splitters.RandomStratifiedSplitter()
  }
  splitter = splitters[split]
  logger.info("About to split dataset with {} splitter.".format(split))
  frac_train = kwargs.get("frac_train", 0.8)
  frac_valid = kwargs.get('frac_valid', 0.1)
  frac_test = kwargs.get('frac_test', 0.1)

  train, valid, test = splitter.train_valid_test_split(
      dataset,
      frac_train=frac_train,
      frac_valid=frac_valid,
      frac_test=frac_test)
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
  return hiv_tasks, (train, valid, test), transformers
