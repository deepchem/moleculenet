import errno
import os
import torch


def decide_metric(dataset):
  if dataset in ['BACE_classification', 'BBBP', 'ClinTox', 'SIDER']:
    return 'roc_auc'
  elif dataset in ['BACE_regression', 'Delaney', 'HOPV', 'Lipo', 'PDBbind']:
    return 'rmse'
  else:
    return ValueError('Unexpected dataset: {}'.format(dataset))


def mkdir_p(path):
  """Create a folder for the given path.

  Parameters
  ----------
  path: str
    Folder to create.
  """
  try:
    os.makedirs(path)
    print('Created directory {}'.format(path))
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      print('Directory {} already exists.'.format(path))
    else:
      raise


def init_trial_path(args):
  """Initialize the path for a hyperparameter setting

  Parameters
  ----------
  args : dict
    Settings

  Returns
  -------
  args : dict
    Settings with the trial path updated
  """
  trial_id = 0
  path_exists = True
  while path_exists:
    trial_id += 1
    path_to_results = args['result_path'] + '/{:d}'.format(trial_id)
    path_exists = os.path.exists(path_to_results)
  mkdir_p(path_to_results)

  return path_to_results


def load_dataset(args):
  splitter = 'scaffold'

  if args['featurizer'] == 'ECFP':
    featurizer = 'ECFP'
  elif args['featurizer'] == 'GC':
    from deepchem.feat import MolGraphConvFeaturizer
    featurizer = MolGraphConvFeaturizer()
  elif args['featurizer'] == 'AC':
    from deepchem.feat import AtomicConvFeaturizer
    featurizer = AtomicConvFeaturizer(
        frag1_num_atoms=100,
        frag2_num_atoms=1000,
        complex_num_atoms=1100,
        max_num_neighbors=12,
        neighbor_cutoff=4)

  if args['featurizer'] in ['flat_combined', 'voxel_combined', 'all_combined', 'ecfp_ligand', 'ecfp_hashed', 'ecfp', 'splif']:
    from deepchem.feat import RdkitGridFeaturizer as RGF
    featurizer = RGF(feature_types=[args['featurizer']], 
      ecfp_power=9, splif_power=5, box_width=6.0, flatten=True, sanitize=True)

  if args['dataset'] == 'BACE_classification':
    from deepchem.molnet import load_bace_classification
    tasks, all_dataset, transformers = load_bace_classification(
        featurizer=featurizer, splitter=splitter, reload=False)
  elif args['dataset'] == 'BBBP':
    from deepchem.molnet import load_bbbp
    tasks, all_dataset, transformers = load_bbbp(
        featurizer=featurizer, splitter=splitter, reload=False)
  elif args['dataset'] == 'BACE_regression':
    from deepchem.molnet import load_bace_regression
    tasks, all_dataset, transformers = load_bace_regression(
        featurizer=featurizer, splitter=splitter, reload=False)
  elif args['dataset'] == 'ClinTox':
    from deepchem.molnet import load_clintox
    tasks, all_dataset, transformers = load_clintox(
        featurizer=featurizer, splitter=splitter, reload=False)
  elif args['dataset'] == 'Delaney':
    from deepchem.molnet import load_delaney
    tasks, all_dataset, transformers = load_delaney(
        featurizer=featurizer, splitter=splitter, reload=False)
  elif args['dataset'] == 'HOPV':
    from deepchem.molnet import load_hopv
    tasks, all_dataset, transformers = load_hopv(
        featurizer=featurizer, splitter=splitter, reload=False)
  elif args['dataset'] == 'SIDER':
    from deepchem.molnet import load_sider
    tasks, all_dataset, transformers = load_sider(
        featurizer=featurizer, splitter=splitter, reload=False)
  elif args['dataset'] == 'Lipo':
    from deepchem.molnet import load_lipo
    tasks, all_dataset, transformers = load_lipo(
        featurizer=featurizer, splitter=splitter, reload=False)
  elif args['dataset'] == 'PDBbind':
    from deepchem.molnet import load_pdbbind
    tasks, all_dataset, transformers = load_pdbbind(
        featurizer=featurizer,
        save_dir='.',
        data_dir='.',
        splitter='random',
        pocket=True,
        set_name='core',  # refined
        reload=False)
  else:
    raise ValueError('Unexpected dataset: {}'.format(args['dataset']))

  return args, tasks, all_dataset, transformers


class EarlyStopper():

  def __init__(self, save_path, metric, patience):
    if metric in ['roc_auc', 'r2']:
      self.best_score = 0
      self.mode = 'higher'
    elif metric in ['rmse']:
      self.best_score = float('inf')
      self.mode = 'lower'
    else:
      raise ValueError('Unexpected metric: {}'.format(metric))

    self.save_path = save_path
    self.max_patience = patience
    self.patience_count = 0

  def __call__(self, model, current_score):
    from deepchem.models import TorchModel
    if self.mode == 'higher' and current_score > self.best_score:
      self.best_score = current_score
      self.patience_count = 0
      if type(model).__bases__[0] == TorchModel:
        torch.save(model.model.state_dict(), self.save_path + '/early_stop.pt')
      else:  # KerasModel
        model.model.save(self.save_path + '/early_stop')
    elif self.mode == 'lower' and current_score < self.best_score:
      self.best_score = current_score
      self.patience_count = 0
      if type(model).__bases__[0] == TorchModel:
        torch.save(model.model.state_dict(), self.save_path + '/early_stop.pt')
      else:  # KerasModel
        model.model.save(self.save_path + '/early_stop')
    else:
      self.patience_count += 1

    return self.patience_count == self.max_patience

  def load_state_dict(self, model):
    model.model.load_state_dict(torch.load(self.save_path + '/early_stop.pt'))

  def load_keras_model(self, model):
    model.restore(model_dir=self.save_path)
