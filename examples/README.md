# MoleculeNet Examples

The repository includes the following example scripts:
- fingerprint
- gnn

## Fingerprint

To train a random forest on 1024-bit ECFP2 fingerprint

```bash
python fingerprint.py
```

The feasible arguments include:
- **Dataset**: `-d dataset`
    - Specifies the dataset to use, which can be one of the following:
        - `BACE_classification`
        - `BACE_regression`
- **Hyperparameter Search (optional)**: `-hs`
    - Perform a hyperparameter search using Bayesian optimization. It determines the best 
      hyperparameters based on the validation metric averaged across 3 runs.
    - If not specified, the script uses default hyperparameters.
- **Number of Hyperparameter Search Trials (optional)**: `-nt num_trials` [default=16]
    - The number of trials for hyperparameter search. This comes into effect only when the user specifies 
      `-hs` as described above.
      
## GNN

To train a GNN using [GraphConv featurization](https://github.com/deepchem/deepchem/blob/master/deepchem/feat/molecule_featurizers/mol_graph_conv_featurizer.py#L89)

```bash
python gnn.py
```

The feasible arguments include:
- **Dataset**: `-d dataset`
    - Specifies the dataset to use, which can be one of the following:
        - `BACE_classification`
        - `BACE_regression`
- **Hyperparameter Search (optional)**: `-hs`
    - Perform a hyperparameter search using Bayesian optimization. It determines the best 
      hyperparameters based on the validation metric averaged across 3 runs.
    - If not specified, the script uses default hyperparameters.
- **Number of Hyperparameter Search Trials (optional)**: `-nt num_trials` [default=16]
    - The number of trials for hyperparameter search. This comes into effect only when the user specifies 
      `-hs` as described above.

## Results

The scripts will save the training results in `results/`, including:
- `configure.json`: the file that stores the training configuration
- `eval.txt`: the file that stores the final validation and test performance

If the scripts perform a hyperparameter search, there will be subfolders under `results/`
corresponding to the search trials. One subfolder per trial. In this case, 
`results/configure.json` and `results/eval.txt` contain the results for the best trial.
