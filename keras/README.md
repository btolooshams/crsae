[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

CRsAE (constrained recurrent sparse auto-encoder)
==============================

Deep residual auto-encoder for expectation maximization-based dictionary learning

Project Organization
------------

    ├── LICENSE
    ├── README.md                       <- The top-level README for developers using this project.
    │
    ├── data
    │   ├── filters
    │   ├── processed
    │   └── raw
    │
    ├── experiments                      <- contains folder for each experiment
    │   └── folder
    │       ├── config
    │       │   ├── config_data.yml
    │       │   └── config_model.yml
    │       │
    │       ├── data
    │       │   ├── H_true.npy           <- only for simulated data
    │       │   └── data.h5
    │       │
    │       ├── reports
    │       │   ├── H_*.pdf
    │       │   ├── H_epochs_*.pdf
    │       │   ├── H_err_*.pdf
    │       │   ├── code_*.pdf
    │       │   ├── denoise_*.pdf
    │       │   ├── loss_*.pdf
    │       │   └── summary_*.yaml
    │       │
    │       └── results
    │           ├── results_lr.h5
    │           ├── results_prediction_*.h5
    │           └── results_training_*.h5
    │
    ├── notebooks
    │
    ├── requirements.txt                 <- The requirements file for reproducing the analysis environment,
    │                                       e.g.(generate by "pipreqs /path/to/project")
    │
    ├── src                              <- Source code for use in this project.
    │   ├── __init__.py
    │   │
    │   ├── callbacks                    <- Custom Keras callbacks.
    │   │   ├── clr_callback.py
    │   │   └── lrfinder_callback.py
    │   │
    │   ├── generators                   <- Functions for generate data and waveforms.
    │   │   ├── make_dataset.py
    │   │   └── process_real_dataset.py
    │   │
    │   ├── layers                       <- custom keras layers.
    │   │   ├── conv_tied_layers.py
    │   │   ├── ista_fista_layers.py
    │   │   ├── trainable_lambda_loss_function_layers.py
    │   │   └── trainable_threshold_relu_layers.py
    │   │
    │   ├── models                       <- build AE models
    │   │   ├── CRsAE.py
    │   │   ├── LCSC.py
    │   │   ├── TLAE.py
    │   │   └── single_layer_autoencoders.py
    │   │
    │   ├── optimizers
    │   │   ├── adam_optimizer.py
    │   │   └── sgd_optimizer.py
    │   │
    │   ├── plotter                      <- Functions to plot                    
    │   │   ├── plot_experiment_results.py
    │   │   └── plot_helpers.py
    │   │
    │   ├── prints                       <- Functions to print
    │   │   └── parameters.py
    │   │
    │   └── run_experiments              <- Functions to run experiments for train and predict.
    │   │   ├── extract_results.py
    │   │   ├── extract_results_helpers.py
    │   │   ├── run_experiment.py
    │   │   ├── run_experiment_find_lr.py
    │   │   └── run_experiment_fwd_various_alpha.py
    │   │
    │   └── trainers
    │       └── trainers.py
    │
    └── tests
