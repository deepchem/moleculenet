# MoleculeNet Leaderboard

## Biophysics

### BACE Classification

| Rank | Model         | Featurization  | Test ROC-AUC     | Validation ROC-AUC | Contact                           | References	                                                                           | Date           |
| ---- | ------------- | -------------- | ---------------- | ------------------ | --------------------------------- | ---------------------------------------------------------------------------------------- | -------------- |
| 1    | Random Forest | 1024-bit ECFP4 | 0.8507 +- 0.0072 | 0.7368 +- 0.0066   | [Mufei Li](mufeili1996@gmail.com) | [Paper](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf), [Code](./examples) | Dec 2nd, 2020  |
| 2    | GCN           | GraphConv      | 0.8175 +- 0.0193 | 0.7430 +- 0.0194   | [Mufei Li](mufeili1996@gmail.com) | [Paper](https://arxiv.org/abs/1609.02907), [Code](./examples)                            | Dec 20th, 2020 |

### BACE Regression

| Rank | Model         | Featurization  | Test RMSE        | Validation RMSE  | Contact                           | References	                                                                             | Date           |
| ---- | ------------- | -------------- | ---------------- | ---------------- | --------------------------------- | ---------------------------------------------------------------------------------------- | -------------- |
| 1    | Random Forest | 1024-bit ECFP4 | 1.3178 +- 0.0081 | 0.6716 +- 0.0059 | [Mufei Li](mufeili1996@gmail.com) | [Paper](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf), [Code](./examples) | Dec 26th, 2020 |
| 2    | GCN           | GraphConv      | 1.6450 +- 0.1325 | 0.5244 +- 0.0200 | [Mufei Li](mufeili1996@gmail.com) | [Paper](https://arxiv.org/abs/1609.02907), [Code](./examples)                            | Dec 26th, 2020 |

## Physiology

### BBBP

| Rank | Model         | Featurization  | Test ROC-AUC     | Validation ROC-AUC | Contact                           | References	                                                                           | Date           |
| ---- | ------------- | -------------- | ---------------- | ------------------ | --------------------------------- | ---------------------------------------------------------------------------------------- | -------------- |
| 1    | Random Forest | 1024-bit ECFP4 | 0.9540 +- 0.0038 | 0.9062 +- 0.0079   | [Mufei Li](mufeili1996@gmail.com) | [Paper](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf), [Code](./examples) | Dec 30th, 2020 |
| 2    | GCN           | GraphConv      | 0.9214 +- 0.0106 | 0.9445 +- 0.0049   | [Mufei Li](mufeili1996@gmail.com) | [Paper](https://arxiv.org/abs/1609.02907), [Code](./examples)                            | Dec 30th, 2020 |
