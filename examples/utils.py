import errno
import os
import torch

def decide_metric(dataset):
  if dataset == 'BACE':
    return 'roc_auc'
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
  from deepchem.molnet import load_bace_classification

  splitter = 'scaffold'
  if args['dataset'] == 'BACE':
    tasks, all_dataset, transformers = load_bace_classification(
        featurizer=args['featurizer'], splitter=splitter, reload=False)
  else:
    raise ValueError('Unexpected dataset: {}'.format(args['dataset']))

  return args, tasks, all_dataset, transformers


class EarlyStopper():
  def __init__(self, save_path, metric, patience):
    if metric in ['roc_auc']:
      self.best_score = 0
      self.mode = 'higher'
    else:
      raise ValueError('Unexpected metric: {}'.format(metric))

    self.save_path = save_path
    self.max_patience = patience
    self.patience_count = 0

  def __call__(self, model, current_score):
    if self.mode == 'higher' and current_score > self.best_score:
      self.best_score = current_score
      self.patience_count = 0
      torch.save(model.model.state_dict(), self.save_path + '/early_stop.pt')
    else:
      self.patience_count += 1

    return self.patience_count == self.max_patience

  def load_state_dict(self, model):
    model.model.load_state_dict(torch.load(self.save_path + '/early_stop.pt'))
