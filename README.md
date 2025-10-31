# Diabetes Linear Model

This project implements Bayesian linear regression with Zellner's g-prior for the Arizona diabetes dataset described in Hoff (2009). It provides tools for fitting the full model as well as performing Bayesian model selection via Gibbs sampling.

## Features
- Robust loader for the `azdiabetes.dat` dataset with automatic parsing of whitespace separated values.
- Closed-form posterior updates for a g-prior linear regression model (part a of the exercise).
- Gibbs sampler for Bayesian model selection and model averaging (part b of the exercise).
- Command line script that runs both analyses and prints posterior summaries.

## Dataset
The original data file is referred to as `azdiabetes.dat` in Hoff (2009). Place a copy of the file at `data/azdiabetes.dat.txt` (or provide a custom path via the CLI). The loader is flexible with respect to whitespace and column separators.

## Usage
The project relies only on the Python standard library, so no additional packages are required. Run the analysis script directly with Python:

```bash
python scripts/run_analysis.py --data-path data/azdiabetes.dat.txt
```

Adjust sampling settings via the optional command line arguments to trade off between accuracy and runtime.

## Testing
Run the automated tests with

```bash
pytest
```

The tests rely on synthetic data to exercise the core algorithms, so no external downloads are required.
