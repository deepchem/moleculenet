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
- **Dataset**: `-d dataset` [default=BACE]
    - Specifies the dataset to use.
    - By default we use `BACE`.
- **Hyperparameter Search (optional)**: `-hs`
    - Perform a hyperparameter search using Bayesian optimization. It determines the best 
      hyperparameters based on the validation metric averaged across 3 runs.
    - If not specified, the script uses default hyperparameters.
- **Number of Hyperparameter Search Trials (optional)**: `-nt num_trials` [default=16]
    - The number of trials for hyperparameter search. This comes into effect only when the user specifies 
      `-hs` as described above.
