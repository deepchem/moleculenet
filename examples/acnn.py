import deepchem as dc
import json
import numpy as np

from copy import deepcopy
from hyperopt import hp, fmin, tpe
from shutil import copyfile

from utils import init_trial_path, load_dataset, EarlyStopper


def load_model(save_path, args, tasks, hyperparams):
  if args['dataset'] in ['PDBbind']:
    mode = 'regression'
  else:
    raise ValueError('Unexpected dataset: {}'.format(args['dataset']))

  if args['featurizer'] == 'AC':
    f1_num_atoms = 100
    f2_num_atoms = 1000
    c_num_atoms = f1_num_atoms + f2_num_atoms
    max_num_neighbors = 12

  if args['model'] == 'ACNN':
    model = dc.models.AtomicConvModel(
        n_tasks=len(tasks),
        frag1_num_atoms=f1_num_atoms,
        frag2_num_atoms=f2_num_atoms,
        complex_num_atoms=c_num_atoms,
        max_num_neighbors=max_num_neighbors,
        layer_sizes=hyperparams['layer_sizes'],
        dropout=hyperparams['dropout'],
        learning_rate=hyperparams['lr'],
        batch_size=12,
        model_dir=save_path,
    )
  else:
    raise ValueError('Unexpected model: {}'.format(args['model']))

  return model


def main(save_path, args, hyperparams):
  # Dataset
  args, tasks, all_dataset, transformers = load_dataset(args)
  train_set, val_set, test_set = all_dataset

  # Metric
  if args['metric'] == 'roc_auc':
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)
  elif args['metric'] == 'rmse':
    metric = dc.metrics.Metric(dc.metrics.rms_score, np.mean)
  elif args['metric'] == 'r2':
    metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)
  else:
    raise ValueError('Unexpected metric: {}'.format(args['metric']))

  all_run_val_metrics = []
  all_run_test_metrics = []

  for _ in range(args['num_runs']):
    # Model
    model = load_model(save_path, args, tasks, hyperparams)
    # Object for early stop tracking
    stopper = EarlyStopper(save_path, args['metric'], args['patience'])

    # 100 for maximum number of epochs
    for epoch in range(100):
      model.fit(
          train_set,
          nb_epoch=1,
          max_checkpoints_to_keep=1,
          deterministic=False,
          restore=epoch > 0)

      val_metric = model.evaluate(val_set, [metric], transformers)
      if args['metric'] == 'roc_auc':
        val_metric = val_metric['mean-roc_auc_score']
      if args['metric'] == 'rmse':
        val_metric = val_metric['mean-rms_score']
      if args['metric'] == 'r2':
        val_metric = val_metric['mean-pearson_r2_score']

      # Early stop
      to_stop = stopper(model, val_metric)
      if to_stop:
        break

    stopper.load_keras_model(model)
    val_metric = model.evaluate(val_set, [metric], transformers)
    test_metric = model.evaluate(test_set, [metric], transformers)

    if args['metric'] == 'roc_auc':
      val_metric = val_metric['mean-roc_auc_score']
      test_metric = test_metric['mean-roc_auc_score']
    elif args['metric'] == 'rmse':
      val_metric = val_metric['mean-rms_score']
      test_metric = test_metric['mean-rms_score']
    elif args['metric'] == 'r2':
      val_metric = val_metric['mean-pearson_r2_score']
      test_metric = test_metric['mean-pearson_r2_score']

    all_run_val_metrics.append(val_metric)
    all_run_test_metrics.append(test_metric)

  with open(save_path + '/eval.txt', 'w') as f:
    f.write('Best val {}: {:.4f} +- {:.4f}\n'.format(
        args['metric'], np.mean(all_run_val_metrics),
        np.std(all_run_val_metrics)))
    f.write('Test {}: {:.4f} +- {:.4f}\n'.format(args['metric'],
                                                 np.mean(all_run_test_metrics),
                                                 np.std(all_run_test_metrics)))

  with open(save_path + '/configure.json', 'w') as f:
    json.dump(hyperparams, f, indent=2)

  return all_run_val_metrics, all_run_test_metrics


def init_hyper_search_space(args):
  # Model-based search space
  if args['model'] == 'ACNN':
    search_space = {
        'lr':
        hp.uniform('lr', low=1e-4, high=3e-1),
        'layer_sizes':
        hp.choice('layer_sizes', [[64, 64, 32], [32, 32, 16], [16, 16, 8]]),
        'dropout':
        hp.uniform('dropout', low=0., high=0.6),
    }
  else:
    raise ValueError('Unexpected model: {}'.format(args['model']))

  return search_space


def bayesian_optimization(args):
  results = []
  candidate_hypers = init_hyper_search_space(args)

  def objective(hyperparams):
    configure = deepcopy(args)
    save_path = init_trial_path(args)
    val_metrics, test_metrics = main(save_path, configure, hyperparams)

    if args['metric'] in ['roc_auc', 'r2']:
      # To maximize a non-negative value is equivalent to minimize its opposite number
      val_metric_to_minimize = -1 * np.mean(val_metrics)
    else:
      val_metric_to_minimize = np.mean(val_metrics)

    results.append((save_path, val_metric_to_minimize, val_metrics,
                    test_metrics))

    return val_metric_to_minimize

  fmin(
      objective,
      candidate_hypers,
      algo=tpe.suggest,
      max_evals=args['num_trials'])
  results.sort(key=lambda tup: tup[1])
  best_trial_path, _, best_val_metrics, best_test_metrics = results[0]

  copyfile(best_trial_path + '/configure.json',
           args['result_path'] + '/configure.json')
  copyfile(best_trial_path + '/eval.txt', args['result_path'] + '/eval.txt')

  return best_val_metrics, best_test_metrics


if __name__ == '__main__':
  import argparse

  from utils import mkdir_p

  parser = argparse.ArgumentParser('Examples for MoleculeNet with ACNN')
  parser.add_argument(
      '-d',
      '--dataset',
      choices=['PDBbind'],
      default='PDBbind',
      help='Dataset to use')
  parser.add_argument(
      '-m',
      '--model',
      choices=['ACNN'],
      default='ACNN',
      help=
      'Options include 1) Atomic Convolutional Neural Network (ACNN) (default: ACNN)'
  )
  parser.add_argument(
      '-f',
      '--featurizer',
      choices=['AC'],
      default='AC',
      help='Options include 1) Atomic Convolution (AC) (default: AC)')
  parser.add_argument(
      '-p',
      '--result-path',
      type=str,
      default='results',
      help='Path to save training results (default: results)')
  parser.add_argument(
      '-r',
      '--num-runs',
      type=int,
      default=3,
      help='Number of runs for each hyperparameter configuration (default: 3)')
  parser.add_argument(
      '-pa',
      '--patience',
      type=int,
      default=30,
      help='Number of epochs to wait before early stop if validation performance '
      'stops getting improved (default: 30)')
  parser.add_argument(
      '-hs',
      '--hyper-search',
      action='store_true',
      help='Whether to perform hyperparameter search '
      'or use the default configuration. (default: False)')
  parser.add_argument(
      '-nt',
      '--num-trials',
      type=int,
      default=16,
      help='Number of trials for hyperparameter search (default: 16)')
  parser.add_argument(
      '-me',
      '--metric',
      type=str,
      choices=['rmse', 'r2'],
      default='rmse',
      help=
      'Validation metric to optimize. Options inclue 1) rmse and 2) r2 (default: rmse)'
  )
  args = parser.parse_args().__dict__

  mkdir_p(args['result_path'])
  if args['hyper_search']:
    print('Start hyperparameter search with Bayesian '
          'optimization for {:d} trials'.format(args['num_trials']))
    val_metrics, test_metrics = bayesian_optimization(args)
  else:
    print('Use the manually specified hyperparameters')
    with open('configures/{}_{}/{}.json'.format(
        args['model'], args['featurizer'], args['dataset'])) as f:
      default_hyperparams = json.load(f)
    val_metrics, test_metrics = main(args['result_path'], args,
                                     default_hyperparams)

    print('Val metric for 3 runs: {:.4f} +- {:.4f}'.format(
        np.mean(val_metrics), np.std(val_metrics)))
    print('Test metric for 3 runs: {:.4f} +- {:.4f}'.format(
        np.mean(test_metrics), np.std(test_metrics)))
