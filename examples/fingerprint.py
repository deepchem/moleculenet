import deepchem as dc
import json
import numpy as np

from copy import deepcopy
from functools import partial
from hyperopt import hp, fmin, tpe
from shutil import copyfile
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from utils import init_trial_path, load_dataset


def rf_model_builder(model_dir, hyperparams, mode):
  if mode == 'classification':
    sklearn_model = RandomForestClassifier(
        n_estimators=hyperparams['n_estimators'],
        criterion=hyperparams['criterion'],
        min_samples_split=hyperparams['min_samples_split'],
        bootstrap=hyperparams['bootstrap'])
  if mode == 'regression':
    sklearn_model = RandomForestRegressor(
        n_estimators=hyperparams['n_estimators'],
        criterion=hyperparams['criterion'],
        min_samples_split=hyperparams['min_samples_split'],
        bootstrap=hyperparams['bootstrap'])
  return dc.models.SklearnModel(sklearn_model, model_dir)


def load_model(args, tasks, hyperparams):
  if args['dataset'] in ['BACE_classification', 'BBBP']:
    mode = 'classification'
  elif args['dataset'] in ['BACE_regression', 'Clearance']:
    mode = 'regression'
  else:
    raise ValueError('Unexpected dataset: {}'.format(args['dataset']))

  if args['model'] == 'RF':
    model = dc.models.SingletaskToMultitask(
        tasks, partial(rf_model_builder, hyperparams=hyperparams, mode=mode))
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
  else:
    raise ValueError('Unexpected metric: {}'.format(args['metric']))

  all_run_val_metrics = []
  all_run_test_metrics = []

  for _ in range(args['num_runs']):
    # Model
    model = load_model(args, tasks, hyperparams)
    model.fit(train_set)

    val_metric = model.evaluate(val_set, [metric], transformers)
    test_metric = model.evaluate(test_set, [metric], transformers)

    if args['metric'] == 'roc_auc':
      val_metric = val_metric['mean-roc_auc_score']
      test_metric = test_metric['mean-roc_auc_score']
    elif args['metric'] == 'rmse':
      val_metric = val_metric['mean-rms_score']
      test_metric = test_metric['mean-rms_score']

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
  if args['model'] == 'RF':
    search_space = {
        'n_estimators': hp.choice('n_estimators', [10, 30, 100]),
        'min_samples_split': hp.choice('min_samples_split', [2, 4, 8, 16, 32]),
        'bootstrap': hp.choice('bootstrap', [True, False]),
    }
    if args['dataset'] in ['BACE_classification', 'BBBP']:
      search_space['criterion'] = hp.choice('criterion', ["gini", "entropy"])
    else:
      search_space['criterion'] = hp.choice('criterion', ["mse", "mae"])
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

    if args['metric'] in ['roc_auc']:
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

  from utils import decide_metric, mkdir_p

  parser = argparse.ArgumentParser('Examples for MoleculeNet with fingerprint')
  parser.add_argument(
      '-d',
      '--dataset',
      choices=['BACE_classification', 'BACE_regression', 'BBBP', 'Clearance'],
      help='Dataset to use')
  parser.add_argument(
      '-m',
      '--model',
      choices=['RF'],
      default='RF',
      help='Options include 1) random forest (RF) (default: RF)')
  parser.add_argument(
      '-f',
      '--featurizer',
      choices=['ECFP'],
      default='ECFP',
      help='Options include 1) ECFP (default: ECFP)')
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
  args = parser.parse_args().__dict__

  # Decide the metric to use based on the dataset
  args['metric'] = decide_metric(args['dataset'])

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
