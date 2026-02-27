# Technical Report: Transferring Machine Learning Models to New Soft Pneumatic Actuators
The goal of this project was to develop calibration methods to make machine learning models more transferable to unseen soft pneumatic actuators.

The design and experiments that we ran are described in detail in the corresponding scientific report.
This technical report focuses on explaining how to reproduce the results from the paper and to serve as a starting point for further development.



## Environment Setup
- Install the [uv package manager](https://docs.astral.sh/uv/) using the following command or follow [the official installation guide](https://docs.astral.sh/uv/getting-started/installation/):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
- Install the project dependencies
```
cd <project_root>
uv sync
```

- update the data dir and output dir in `./src/conf/config.yaml` to match the path on your file system.


## Repository Structure
At the top level this repository is organized into:
- `./analysis_data/` -> contains the dataframes that are the basis for the analysis and plots
- `./plots/` -> all plots features in the scientific report
- `./src/` -> the source code
- `./uv.lock` and `./pyproject.toml` -> uv dependency management
- `./run_mlfow.sh` -> shortcut script to start the MLFLow server
- `./README.md` -> this file
- `./data/` -> the audio files for the different datasets

In the source folder, there are the following files:
- `./src/main.py` -> run the training of the ml pipeline
- `./src/morphembed.py` -> the training run logic implemented in a torch lightning module
- `./src/finger_scaling_laws.py` -> run the finger scaling law experiment
- `./src/fit_affine_transforms.py.py` -> run the affine transformation experiment
- `./src/evaluation.py` -> the transferability evaluation that is called from the pipeline and some scripts 
- `./src/create_plots.py` -> script to create all plots
- `./src/datamodule.py` and `./src/datasets.py` -> everything related to datasets and dataloading
- `./src/losses.py` -> the loss functions
- `./src/models.py` -> the neural network definitions
- `./src/utils.py` and `./src/preprocessing.py` -> different utility functions around initial dataloading, fourier transform and others
- `./src/explore_automl.py` -> exploratory AutoML script

The configuration is done using [hydra.cc](https://hydra.cc/) in the `./src/conf/` folder:
- `conf/config.yaml` -> the main config, contains the default values for the subconfigs in the subfolders and which datasets to include.
- `conf/data/*` -> contains the different configuration (raw, stft, ft) for loading the audio (e.g. window size, stft transform)
- `conf/datasets/*` -> the dataset configuration, contains the finger identifier, a short description and which files to include.
- `conf/experiment` -> pre-configured experiments to run, e.g. to generate data normalized in different ways
- `conf/model/*` -> the available models and their parameter, e.g. latent dimensions or sizes of hidden layers
- `conf/training/*` -> the training hyperparameter configuration with e.g. batch_size, max_epochs, etc.

## Datasets
All the data is inside `./data/` with each dataset in different subfolder.
Within each dataset folder, the following file can be found:
- `0_sweep.wav` which is the active sound played.
- Files like `100_none.wav` or `17_7.5.wav` where the first part before the underscore is the sample id, and the second part "none" or "7.5" are the labels.
- ".pkl" files for models (can be ignored)
The regression and classification datasets have many samples of one second length,
the unlabeled datasets contain only one long sample (10 mins) with the active sound being repeatedly played for one second.

The datasets and tasks are described in more detail in the scientific report.


## Starting a Training Run for the Neural Network
To start a training run without further configuration, you can run:
```bash
uv run src/main.py
```
To override the configuration, you can either change the files in `./src/conf` or use [hydra overrides](https://hydra.cc/docs/intro/)

## Reproducing the plots:
To reproduce the plots with the data already in the repository, run the following command:
```bash
uv run src/create_plots.py
```
The plots are saved to `./plots`


## Recreating the Analysis Datasets
First, we need to create the z-score normalized and stft transformed data, this is required to be able to run the affine transformation and finger scaling law experiments.

###  Create the Z-Score Normalized STFT Transformed Data
Make sure MLFlow is running and you followed the environment setup guide.
```bash
sh run_mlfow.sh
```
- Run this command:
```
uv run src/main.py  +experiment=no_normalisation
```
- Go to the MLFlow Dashboard at `localhost:5000` 
- Navigate to the last run in the default experiment
- Go to the artifacts tab
- Download the `all_embeddings.parquet` file
- Move it to `<project_root>/analysis_data/all_embeddings.parquet`

### Normalization Evaluation Datasets
For creating the evaluation results for the different normalizations, follow these steps:
- Make sure MLFlow is running and you followed the environment setup guide.
```bash
sh run_mlfow.sh
```
- Create the different norm datasets by running (at the project root):
```bash
uv run src/main.py --multirun +experiment=no_normalisation,min-max_normalisation,z-score_normalisation.yaml
```
- Go to the MLFlow Dashboard at `localhost:5000` and navigate to the default experiment
- For each of the last three runs:
    - Open the run and check the description, it should contain one of `no_norm`, `z-score_normalisation`, or `min-max_normalisation`.
    - Go to the artifacts tab and download the `test_results_regression.json` file
    - Move it to `<project_root>/analysis_data/test_results_regression_{no_norm|min-max|z-score}.json` and pick the appropriate name 
        from the curly brackets based on what you found in the description, e.g. `test_results_regression_no_norm.json`
The data is now saved in the appropriate place and can be plotted.

### Finger Scaling Law Dataset
Make sure you created the z-score normalized stft transformed dataset first (see section **Create the Z-Score Normalized STFT Transformed Data**).

Navigate to the project root and run:
```bash
uv run src/finger_scaling_laws.py
```
The output is saved to `<project_root>/analysis_data/finger_scaling_laws.parquet` and can now be plotted.


### Affine Transformations Dataset
Make sure you created the z-score normalized stft transformed dataset first (see section **Create the Z-Score Normalized STFT Transformed Data**).

Navigate to the project root and run:
```bash
uv run src/fit_affine_transforms.py
```
The output is saved to `<project_root>/analysis_data/affine_transformations_results.parquet` and can now be plotted.
