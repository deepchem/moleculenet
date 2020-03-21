"""
SIDER dataset loader.
"""
import os
import logging
import moleculenet 

logger = logging.getLogger(__name__)

DEFAULT_DIR = moleculenet.utils.get_data_dir()
SIDER_URL = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/sider.csv.gz'


def load_sider(featurizer='ECFP',
               split='index',
               reload=True,
               K=4,
               data_dir=None,
               save_dir=None,
               **kwargs):
  logger.info("About to load SIDER dataset.")
  if data_dir is None:
    data_dir = DEFAULT_DIR
  if save_dir is None:
    save_dir = DEFAULT_DIR

  if reload:
    save_folder = os.path.join(save_dir, "sider-featurized", str(featurizer))
    if featurizer == "smiles2img":
      img_spec = kwargs.get("img_spec", "std")
      save_folder = os.path.join(save_folder, img_spec)
    save_folder = os.path.join(save_folder, str(split))

  dataset_file = os.path.join(data_dir, "sider.csv.gz")
  if not os.path.exists(dataset_file):
    moleculenet.utils.download_url(url=SIDER_URL, dest_dir=data_dir)

  dataset = moleculenet.utils.load_from_disk(dataset_file)
  logger.info("Columns of dataset: %s" % str(dataset.columns.values))
  logger.info("Number of examples in dataset: %s" % str(dataset.shape[0]))
  SIDER_tasks = dataset.columns.values[1:].tolist()

  if reload:
    loaded, all_dataset, transformers = moleculenet.utils.load_dataset_from_disk(
        save_folder)
    if loaded:
      return SIDER_tasks, all_dataset, transformers

  # Featurize SIDER dataset
  logger.info("About to featurize SIDER dataset.")
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

  logger.info("SIDER tasks: %s" % str(SIDER_tasks))
  logger.info("%d tasks in total" % len(SIDER_tasks))

  loader = moleculenet.data.CSVLoader(
      tasks=SIDER_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file)
  logger.info("%d datapoints in SIDER dataset" % len(dataset))

  # Initialize transformers
  transformers = [
      moleculenet.transformers.BalancingTransformer(transform_w=True, dataset=dataset)
  ]
  logger.info("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  if split == None:
    return SIDER_tasks, (dataset, None, None), transformers

  splitters = {
      'index': moleculenet.splitters.IndexSplitter(),
      'random': moleculenet.splitters.RandomSplitter(),
      'scaffold': moleculenet.splitters.ScaffoldSplitter(),
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
    if reload:
      moleculenet.utils.save_dataset_to_disk(save_folder, train, valid, test,
                                               transformers)
    all_dataset = (train, valid, test)
  return SIDER_tasks, all_dataset, transformers
