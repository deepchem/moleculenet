import deepchem as dc
import json
import numpy as np

from copy import deepcopy
from functools import partial
from hyperopt import hp, fmin, tpe
from shutil import copyfile
from sklearn.ensemble import RandomForestClassifier

from utils import mkdir_p, init_trial_path


def decide_metric(dataset):
    if dataset == 'BACE':
        return 'roc_auc'
    else:
        return ValueError('Unexpected dataset: {}'.format(dataset))


def load_dataset(args):
    from deepchem.molnet import load_bace_classification

    splitter = 'scaffold'
    if args['dataset'] == 'BACE':
        tasks, all_dataset, transformers = load_bace_classification(
            featurizer=args['featurizer'], splitter=splitter)
    else:
        raise ValueError('Unexpected dataset: {}'.format(args['dataset']))

    return args, tasks, all_dataset, transformers


def rf_model_builder(model_dir, hyperparams):
    sklearn_model = RandomForestClassifier(
        n_estimators=hyperparams['n_estimators'],
        criterion=hyperparams['criterion'],
        min_samples_split=hyperparams['min_samples_split'],
        bootstrap=hyperparams['bootstrap'])
    return dc.models.SklearnModel(sklearn_model, model_dir)


def load_model(args, tasks, hyperparams):
    if args['model'] == 'RF':
        model = dc.models.SingletaskToMultitask(
            tasks, partial(rf_model_builder, hyperparams=hyperparams))
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

        all_run_val_metrics.append(val_metric)
        all_run_test_metrics.append(test_metric)

    with open(save_path + '/eval.txt', 'w') as f:
        f.write('Best val {}: {:.4f} +- {:.4f}\n'.format(
            args['metric'], np.mean(all_run_val_metrics),
            np.std(all_run_val_metrics)))
        f.write('Test {}: {:.4f} +- {:.4f}\n'.format(
            args['metric'], np.mean(all_run_test_metrics),
            np.std(all_run_test_metrics)))

    with open(save_path + '/configure.json', 'w') as f:
        json.dump(hyperparams, f, indent=2)

    return all_run_val_metrics, all_run_test_metrics


def init_hyper_search_space(args):
    # Model-based search space
    if args['model'] == 'RF':
        search_space = {
            'n_estimators':
            hp.choice('n_estimators', [10, 30, 100]),
            'criterion':
            hp.choice('criterion', ["gini", "entropy"]),
            'min_samples_split':
            hp.choice('min_samples_split', [2, 4, 8, 16, 32]),
            'bootstrap':
            hp.choice('bootstrap', [True, False]),
        }
    else:
        raise ValueError('Unexpected model: {}'.format(args['model']))

    # Feature-based search space
    search_space.update({
        'radius': [1, 2, 3, 4, 5],
        'size': [512, 1024, 2048],
        'chiral': [True, False],
        'bonds': [True, False],
        'features': [True, False]
    })

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

    parser = argparse.ArgumentParser(
        'Examples for MoleculeNet with fingerprint')
    parser.add_argument(
        '-d',
        '--dataset',
        choices=['BACE'],
        default='BACE',
        help='Dataset to use (default: BACE)')
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
        help='Number of runs for each hyperparameter configuration (default: 3)'
    )
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
        default_hyperparams = {
            'bootstrap': True,
            'criterion': "entropy",
            'min_samples_split': 32,
            'n_estimators': 30,
        }
        val_metrics, test_metrics = main(args['result_path'], args,
                                         default_hyperparams)

    print('Val metric for 3 runs: {:.4f} +- {:.4f}'.format(
        np.mean(val_metrics), np.std(val_metrics)))
    print('Test metric for 3 runs: {:.4f} +- {:.4f}'.format(
        np.mean(test_metrics), np.std(test_metrics)))
