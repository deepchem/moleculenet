"""
Simple utils.
"""

import joblib
import numpy as np

def save_to_disk(dataset, filename, compress=3):
  """Save a dataset to joblib file."""
  joblib.dump(dataset, filename, compress=compress)

def log(string, verbose=True):
  """Print string if verbose."""
  if verbose:
    print(string)

# TODO(rbharath): We should move this in with DiskDataset and make private
def save_metadata(tasks, metadata_df, data_dir):
  """
  Saves the metadata for a DiskDataset
  Parameters
  ----------
  tasks: list of str
    Tasks of DiskDataset
  metadata_df: pd.DataFrame
  data_dir: str
    Directory to store metadata
  Returns
  -------
  """
  if isinstance(tasks, np.ndarray):
    tasks = tasks.tolist()
  metadata_filename = os.path.join(data_dir, "metadata.csv.gzip")
  tasks_filename = os.path.join(data_dir, "tasks.json")
  with open(tasks_filename, 'w') as fout:
    json.dump(tasks, fout)
  metadata_df.to_csv(metadata_filename, index=False, compression='gzip')

# TODO(rbharath): Should this also be private for DiskDataset?
def load_from_disk(filename):
  """Load a dataset from file."""
  name = filename
  if os.path.splitext(name)[1] == ".gz":
    name = os.path.splitext(name)[0]
  if os.path.splitext(name)[1] == ".pkl":
    return load_pickle_from_disk(filename)
  elif os.path.splitext(name)[1] == ".joblib":
    return joblib.load(filename)
  elif os.path.splitext(name)[1] == ".csv":
    # First line of user-specified CSV *must* be header.
    df = pd.read_csv(filename, header=0)
    df = df.replace(np.nan, str(""), regex=True)
    return df
  else:
    raise ValueError("Unrecognized filetype for %s" % filename)

def pad_array(x, shape, fill=0, both=False):
  """
  Pad an array with a fill value.

  Parameters
  ----------
  x : ndarray
      Matrix.
  shape : tuple or int
      Desired shape. If int, all dimensions are padded to that size.
  fill : object, optional (default 0)
      Fill value.
  both : bool, optional (default False)
      If True, split the padding on both sides of each axis. If False,
      padding is applied to the end of each axis.
  """
  x = np.asarray(x)
  if not isinstance(shape, tuple):
    shape = tuple(shape for _ in range(x.ndim))
  pad = []
  for i in range(x.ndim):
    diff = shape[i] - x.shape[i]
    assert diff >= 0
    if both:
      a, b = divmod(diff, 2)
      b += a
      pad.append((a, b))
    else:
      pad.append((0, diff))
  pad = tuple(pad)
  x = np.pad(x, pad, mode='constant', constant_values=fill)
  return x
