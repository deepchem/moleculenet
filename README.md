# MoleculeNet Leaderboard

## Biophysics

### BACE

| Rank | Model         | Featurization  | Test ROC-AUC     | Validation ROC-AUC | Contact                           | References	                                                                           | Date           |
| ---- | ------------- | -------------- | ---------------- | ------------------ | --------------------------------- | ---------------------------------------------------------------------------------------- | -------------- |
| 1    | Random Forest | 1024-bit ECFP2 | 0.8507 +- 0.0072 | 0.7368 +- 0.0066   | [Mufei Li](mufeili1996@gmail.com) | [Paper](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf), [Code](./examples) | Dec 2nd, 2020  |
| 2    | GCN           | GraphConv      | 0.8175 +- 0.0193 | 0.7430 +- 0.0194   | [Mufei Li](mufeili1996@gmail.com) | [Paper](https://arxiv.org/abs/1609.02907), [Code](./examples)                            | Dec 20th, 2020 |
