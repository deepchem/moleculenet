# MoleculeNet Leaderboard

## Physical Chemistry

### Delaney (ESOL)

| Rank | Model         | Featurization  | Test RMSE        | Validation RMSE  | Contact                           | References	                                                                             | Date           |
| ---- | ------------- | -------------- | ---------------- | ---------------- | --------------------------------- | ---------------------------------------------------------------------------------------- | -------------- |
| 1    | GCN           | GraphConv      | 0.8851 +- 0.0292 | 0.9405 +- 0.0310 | [Mufei Li](mufeili1996@gmail.com) | [Paper](https://arxiv.org/abs/1609.02907), [Code](./examples)                            | Jan 28th, 2020 |
| 2    | Random Forest | 1024-bit ECFP4 | 1.7406 +- 0.0261 | 1.7932 +- 0.0153 | [Mufei Li](mufeili1996@gmail.com) | [Paper](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf), [Code](./examples) | Jan 28th, 2020 |

### Lipo

| Rank | Model         | Featurization  | Test RMSE        | Validation RMSE  | Contact                           | References	                                                                             | Date           |
| ---- | ------------- | -------------- | ---------------- | ---------------- | --------------------------------- | ---------------------------------------------------------------------------------------- | -------------- |
| 1    | GCN           | GraphConv      | 0.7806 +- 0.0404 | 0.7897 +- 0.0553   | [Yuanqi Du](ydu6@gmu.edu) | [Paper](https://arxiv.org/abs/1609.02907), [Code](./examples)                            | Feb 13th, 2021 |
| 2    | Random Forest | 1024-bit ECFP4 | 0.9621 +- 0.0030 | 1.0024 +- 0.0029   | [Yuanqi Du](ydu6@gmu.edu) | [Paper](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf), [Code](./examples) | Feb 13th, 2021 |

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

### PDBbind (core)

| Rank | Model         | Featurization  | Test RMSE (R^2)     | Validation RMSE (R^2) | Contact                           | References                                                                               | Date           |
| ---- | ------------- | -------------- | ---------------- | ------------------ | --------------------------------- | ---------------------------------------------------------------------------------------- | -------------- |
| 1    | ACNN | AtomicConv | 1.8604 +- 0.0540 (0.5405 +- 0.0124) | 1.5885 +- 0.0865 (0.6015 +- 0.0330)   | [Nathan Frey](n.frey@seas.upenn.edu) | [Paper](https://arxiv.org/pdf/1703.10603.pdf), [Code](./examples) | March 24, 2021  |
| 2    | RF | Ligand ECFP | 1.9548 +- 0.0372 (0.1763 +- 0.0802) | 2.0035 +- 0.0441 (0.2662 +- 0.0477)   | [Nathan Frey](n.frey@seas.upenn.edu) | [Paper](https://pubs.acs.org/doi/pdf/10.1021/ci100050t), [Code](./examples) | April 15, 2021  |
| 3    | MLP | Ligand ECFP | 1.9403 +- 0.0105 (0.0881 +- 0.0606) | 1.6162 +- 0.0115 (0.1371 +- 0.1173)   | [Nathan Frey](n.frey@seas.upenn.edu) | [Paper](https://pubs.acs.org/doi/pdf/10.1021/ci100050t), [Code](./examples) | April 15, 2021  | 
| 4    | MLP | SPLIF | 2.6994 +- 0.0879 (0.0293 +- 0.0274) | 2.1859 +- 0.1000 (0.0866 +- 0.0008)   | [Nathan Frey](n.frey@seas.upenn.edu) | [Paper](https://pubs.acs.org/doi/abs/10.1021/ci500319f), [Code](./examples) | April 15, 2021  | 
| 5    | RF | Voxelized ECFP | 2.5717 +- 0.0302 (0.0117 +- 0.0027) | 2.3351 +- 0.0241 (0.0405 +- 0.0099)   | [Nathan Frey](n.frey@seas.upenn.edu) | [Paper](https://pubs.acs.org/doi/pdf/10.1021/ci100050t), [Code](./examples) | April 15, 2021  | 

## Physiology

### BBBP

| Rank | Model         | Featurization  | Test ROC-AUC     | Validation ROC-AUC | Contact                           | References	                                                                           | Date           |
| ---- | ------------- | -------------- | ---------------- | ------------------ | --------------------------------- | ---------------------------------------------------------------------------------------- | -------------- |
| 1    | Random Forest | 1024-bit ECFP4 | 0.9540 +- 0.0038 | 0.9062 +- 0.0079   | [Mufei Li](mufeili1996@gmail.com) | [Paper](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf), [Code](./examples) | Dec 30th, 2020 |
| 2    | GCN           | GraphConv      | 0.9214 +- 0.0106 | 0.9445 +- 0.0049   | [Mufei Li](mufeili1996@gmail.com) | [Paper](https://arxiv.org/abs/1609.02907), [Code](./examples)                            | Dec 30th, 2020 |

### ClinTox

| Rank | Model         | Featurization  | Test ROC-AUC     | Validation ROC-AUC | Contact                           | References	                                                                           | Date           |
| ---- | ------------- | -------------- | ---------------- | ------------------ | --------------------------------- | ---------------------------------------------------------------------------------------- | -------------- |
| 1    | GCN           | GraphConv      | 0.9065 +- 0.0179 | 0.9880 +- 0.0073   | [Mufei Li](mufeili1996@gmail.com) | [Paper](https://arxiv.org/abs/1609.02907), [Code](./examples)                            | Jan 22nd, 2021 |
| 2    | Random Forest | 1024-bit ECFP4 | 0.7829 +- 0.0235 | 0.8883 +- 0.0230   | [Mufei Li](mufeili1996@gmail.com) | [Paper](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf), [Code](./examples) | Jan 22nd, 2021 |

### SIDER

| Rank | Model         | Featurization  | Test ROC-AUC     | Validation ROC-AUC | Contact                           | References	                                                                           | Date           |
| ---- | ------------- | -------------- | ---------------- | ------------------ | --------------------------------- | ---------------------------------------------------------------------------------------- | -------------- |
| 1    | Random Forest | 1024-bit ECFP4 | 0.6504 +- 0.0011 | 0.6197 +- 0.0023   | [Yuanqi Du](ydu6@gmu.edu) | [Paper](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf), [Code](./examples) | Feb 10th, 2021 |
| 2    | GCN           | GraphConv      | 0.6265 +- 0.0076 | 0.6409 +- 0.0094   | [Yuanqi Du](ydu6@gmu.edu) | [Paper](https://arxiv.org/abs/1609.02907), [Code](./examples)                            | Feb 10th, 2021 |

## Materials Science

### HOPV

| Rank | Model         | Featurization  | Test RMSE        | Validation RMSE  | Contact                           | References	                                                                             | Date           |
| ---- | ------------- | -------------- | ---------------- | ---------------- | --------------------------------- | ---------------------------------------------------------------------------------------- | -------------- |
| 1    | Random Forest | 1024-bit ECFP4 | 3.7283 +- 0.0219 | 2.7107 +- 0.0173 | [Mufei Li](mufeili1996@gmail.com) | [Paper](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf), [Code](./examples) | Feb 4th, 2021  | 
| 2    | GCN           | GraphConv      | 5.4312 +- 2.4279 | 2.6516 +- 0.1190 | [Mufei Li](mufeili1996@gmail.com) | [Paper](https://arxiv.org/abs/1609.02907), [Code](./examples)                            | Feb 4th, 2021  |

