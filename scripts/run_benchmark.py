# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 14:25:40 2017

@author: Zhenqin Wu
"""
import os
import time
import csv
import numpy as np
import tensorflow as tf
import deepchem
import pickle
from deepchem.molnet.run_benchmark_models import benchmark_classification, benchmark_regression
from deepchem.molnet.check_availability import CheckFeaturizer, CheckSplit
from deepchem.molnet.preset_hyper_parameters import hps


def run_benchmark(datasets,
                  model,
                  split=None,
                  metric=None,
                  direction=True,
                  featurizer=None,
                  n_features=0,
                  out_path='.',
                  hyper_parameters=None,
                  hyper_param_search=False,
                  max_iter=20,
                  search_range=2,
                  test=False,
                  reload=True,
                  seed=123):
  """
  Run benchmark test on designated datasets with deepchem(or user-defined) model

  Parameters
  ----------
  datasets: list of string
      choice of which datasets to use, should be: bace_c, bace_r, bbbp, chembl,
      clearance, clintox, delaney, hiv, hopv, kaggle, lipo, muv, nci, pcba,
      pdbbind, ppb, qm7, qm7b, qm8, qm9, sampl, sider, tox21, toxcast, uv, factors,
      kinase
  model: string or user-defined model stucture
      choice of which model to use, deepchem provides implementation of
      logistic regression, random forest, multitask network,
      bypass multitask network, irv, graph convolution;
      for user define model, it should include function: fit, evaluate
  split: string,  optional (default=None)
      choice of splitter function, None = using the default splitter
  metric: string, optional (default=None)
      choice of evaluation metrics, None = using the default metrics(AUC & R2)
  direction: bool, optional(default=True)
      Optimization direction when doing hyperparameter search
      Maximization(True) or minimization(False)
  featurizer: string or dc.feat.Featurizer,  optional (default=None)
      choice of featurization, None = using the default corresponding to model
      (string only applicable to deepchem models)
  n_features: int, optional(default=0)
      depending on featurizers, redefined when using deepchem featurizers,
      need to be specified for user-defined featurizers(if using deepchem models)
  out_path: string, optional(default='.')
      path of result file
  hyper_parameters: dict, optional (default=None)
      hyper parameters for designated model, None = use preset values
  hyper_param_search: bool, optional(default=False)
      whether to perform hyper parameter search, using gaussian process by default
  max_iter: int, optional(default=20)
      number of optimization trials
  search_range: int(float), optional(default=4)
      optimization on [initial values / search_range,
                       initial values * search_range]
  test: boolean, optional(default=False)
      whether to evaluate on test set
  reload: boolean, optional(default=True)
      whether to save and reload featurized datasets
  """
  for dataset in datasets:
    if dataset in [
        'bace_c', 'bbbp', 'clintox', 'hiv', 'muv', 'pcba', 'pcba_146',
        'pcba_2475', 'sider', 'tox21', 'toxcast'
    ]:
      mode = 'classification'
      if metric == None:
        metric = [
            deepchem.metrics.Metric(deepchem.metrics.roc_auc_score, np.mean),
        ]
    elif dataset in [
        'bace_r', 'chembl', 'clearance', 'delaney', 'hopv', 'kaggle', 'lipo',
        'nci', 'pdbbind', 'ppb', 'qm7', 'qm7b', 'qm8', 'qm9', 'sampl',
        'thermosol'
    ]:
      mode = 'regression'
      if metric == None:
        metric = [
            deepchem.metrics.Metric(deepchem.metrics.pearson_r2_score, np.mean)
        ]
    else:
      raise ValueError('Dataset not supported')

    if featurizer == None and isinstance(model, str):
      # Assigning featurizer if not user defined
      pair = (dataset, model)
      if pair in CheckFeaturizer:
        featurizer = CheckFeaturizer[pair][0]
        n_features = CheckFeaturizer[pair][1]
      else:
        continue

    if not split in [None] + CheckSplit[dataset]:
      continue

    loading_functions = {
        'bace_c': deepchem.molnet.load_bace_classification,
        'bace_r': deepchem.molnet.load_bace_regression,
        'bbbp': deepchem.molnet.load_bbbp,
        'chembl': deepchem.molnet.load_chembl,
        'clearance': deepchem.molnet.load_clearance,
        'clintox': deepchem.molnet.load_clintox,
        'delaney': deepchem.molnet.load_delaney,
        'factors': deepchem.molnet.load_factors,
        'hiv': deepchem.molnet.load_hiv,
        'hopv': deepchem.molnet.load_hopv,
        'hppb': deepchem.molnet.load_hppb,
        'kaggle': deepchem.molnet.load_kaggle,
        'kinase': deepchem.molnet.load_kinase,
        'lipo': deepchem.molnet.load_lipo,
        'muv': deepchem.molnet.load_muv,
        'nci': deepchem.molnet.load_nci,
        'pcba': deepchem.molnet.load_pcba,
        'pcba_146': deepchem.molnet.load_pcba_146,
        'pcba_2475': deepchem.molnet.load_pcba_2475,
        'pdbbind': deepchem.molnet.load_pdbbind_grid,
        'ppb': deepchem.molnet.load_ppb,
        'qm7': deepchem.molnet.load_qm7_from_mat,
        'qm7b': deepchem.molnet.load_qm7b_from_mat,
        'qm8': deepchem.molnet.load_qm8,
        'qm9': deepchem.molnet.load_qm9,
        'sampl': deepchem.molnet.load_sampl,
        'sider': deepchem.molnet.load_sider,
        'thermosol': deepchem.molnet.load_thermosol,
        'tox21': deepchem.molnet.load_tox21,
        'toxcast': deepchem.molnet.load_toxcast,
        'uv': deepchem.molnet.load_uv,
    }

    print('-------------------------------------')
    print('Benchmark on dataset: %s' % dataset)
    print('-------------------------------------')
    # loading datasets
    if split is not None:
      print('Splitting function: %s' % split)
      tasks, all_dataset, transformers = loading_functions[dataset](
          featurizer=featurizer, split=split, reload=reload)
    else:
      tasks, all_dataset, transformers = loading_functions[dataset](
          featurizer=featurizer, reload=reload)

    train_dataset, valid_dataset, test_dataset = all_dataset

    time_start_fitting = time.time()
    train_score = {}
    valid_score = {}
    test_score = {}

    if hyper_param_search:
      if hyper_parameters is None:
        hyper_parameters = hps[model]
      search_mode = deepchem.hyper.GaussianProcessHyperparamOpt(model)
      hyper_param_opt, _ = search_mode.hyperparam_search(
          hyper_parameters,
          train_dataset,
          valid_dataset,
          transformers,
          metric,
          direction=direction,
          n_features=n_features,
          n_tasks=len(tasks),
          max_iter=max_iter,
          search_range=search_range)
      hyper_parameters = hyper_param_opt
    if isinstance(model, str):
      if mode == 'classification':
        train_score, valid_score, test_score = benchmark_classification(
            train_dataset,
            valid_dataset,
            test_dataset,
            tasks,
            transformers,
            n_features,
            metric,
            model,
            test=test,
            hyper_parameters=hyper_parameters,
            seed=seed)
      elif mode == 'regression':
        train_score, valid_score, test_score = benchmark_regression(
            train_dataset,
            valid_dataset,
            test_dataset,
            tasks,
            transformers,
            n_features,
            metric,
            model,
            test=test,
            hyper_parameters=hyper_parameters,
            seed=seed)
    else:
      model.fit(train_dataset)
      train_score['user_defined'] = model.evaluate(train_dataset, metric,
                                                   transformers)
      valid_score['user_defined'] = model.evaluate(valid_dataset, metric,
                                                   transformers)
      if test:
        test_score['user_defined'] = model.evaluate(test_dataset, metric,
                                                    transformers)

    time_finish_fitting = time.time()

    with open(os.path.join(out_path, 'results.csv'), 'a') as f:
      writer = csv.writer(f)
      model_name = list(train_score.keys())[0]
      for i in train_score[model_name]:
        output_line = [
            dataset,
            str(split), mode, model_name, i, 'train',
            train_score[model_name][i], 'valid', valid_score[model_name][i]
        ]
        if test:
          output_line.extend(['test', test_score[model_name][i]])
        output_line.extend(
            ['time_for_running', time_finish_fitting - time_start_fitting])
        writer.writerow(output_line)
    if hyper_param_search:
      with open(os.path.join(out_path, dataset + model + '.pkl'), 'w') as f:
        pickle.dump(hyper_parameters, f)


#
# Note by @XericZephyr. Reason why I spun off this function:
#   1. Some model needs dataset information.
#   2. It offers us possibility to **cache** the dataset
#      if the featurizer runs very slow, e.g., GraphConv.
#   2+. The cache can even happen at Travis CI to accelerate
#       CI testing.
#
def load_dataset(dataset, featurizer, split='random'):
  """
  Load specific dataset for benchmark.

  Parameters
  ----------
  dataset: string
      choice of which datasets to use, should be: tox21, muv, sider,
      toxcast, pcba, delaney, factors, hiv, hopv, kaggle, kinase, nci,
      clintox, hiv, pcba_128, pcba_146, pdbbind, chembl, qm7, qm7b, qm9,
      sampl, uv
  featurizer: string or dc.feat.Featurizer.
      choice of featurization.
  split: string,  optional (default=None)
      choice of splitter function, None = using the default splitter
  """
  dataset_loading_functions = {
      'bace_c': deepchem.molnet.load_bace_classification,
      'bace_r': deepchem.molnet.load_bace_regression,
      'bbbp': deepchem.molnet.load_bbbp,
      'chembl': deepchem.molnet.load_chembl,
      'clearance': deepchem.molnet.load_clearance,
      'clintox': deepchem.molnet.load_clintox,
      'delaney': deepchem.molnet.load_delaney,
      'factors': deepchem.molnet.load_factors,
      'hiv': deepchem.molnet.load_hiv,
      'hopv': deepchem.molnet.load_hopv,
      'hppb': deepchem.molnet.load_hppb,
      'kaggle': deepchem.molnet.load_kaggle,
      'kinase': deepchem.molnet.load_kinase,
      'lipo': deepchem.molnet.load_lipo,
      'muv': deepchem.molnet.load_muv,
      'nci': deepchem.molnet.load_nci,
      'pcba': deepchem.molnet.load_pcba,
      'pcba_128': deepchem.molnet.load_pcba_128,
      'pcba_146': deepchem.molnet.load_pcba_146,
      'pcba_2475': deepchem.molnet.load_pcba_2475,
      'pdbbind': deepchem.molnet.load_pdbbind_grid,
      'ppb': deepchem.molnet.load_ppb,
      'qm7': deepchem.molnet.load_qm7_from_mat,
      'qm7b': deepchem.molnet.load_qm7b_from_mat,
      'qm8': deepchem.molnet.load_qm8,
      'qm9': deepchem.molnet.load_qm9,
      'sampl': deepchem.molnet.load_sampl,
      'sider': deepchem.molnet.load_sider,
      'thermosol': deepchem.molnet.load_thermosol,
      'tox21': deepchem.molnet.load_tox21,
      'toxcast': deepchem.molnet.load_toxcast,
      'uv': deepchem.molnet.load_uv
  }
  print('-------------------------------------')
  print('Loading dataset: %s' % dataset)
  print('-------------------------------------')
  # loading datasets
  if split is not None:
    print('Splitting function: %s' % split)
  tasks, all_dataset, transformers = dataset_loading_functions[dataset](
      featurizer=featurizer, split=split)
  return tasks, all_dataset, transformers


def benchmark_model(model, all_dataset, transformers, metric, test=False):
  """
  Benchmark custom model.

  model: user-defined model stucture
    For user define model, it should include function: fit, evaluate.

  all_dataset: (train, test, val) data tuple.
    Returned by `load_dataset` function.

  transformers

  metric: string
    choice of evaluation metrics.


  """
  time_start_fitting = time.time()
  train_score = .0
  valid_score = .0
  test_score = .0

  train_dataset, valid_dataset, test_dataset = all_dataset

  model.fit(train_dataset)
  train_score = model.evaluate(train_dataset, metric, transformers)
  valid_score = model.evaluate(valid_dataset, metric, transformers)
  if test:
    test_score = model.evaluate(test_dataset, metric, transformers)

  time_finish_fitting = time.time()
  time_for_running = time_finish_fitting - time_start_fitting

  return train_score, valid_score, test_score, time_for_running
