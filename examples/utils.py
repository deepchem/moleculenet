import errno
import os


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
