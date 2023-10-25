This repository contains code corresponding to the manuscript "Power and reproducibility in the external validation of brain-phenotype predictions" by Matthew Rosenblatt, Link Tejavibulya, Chris Camp, Rongtao Jiang, Margaret L. Westwater, Stephanie Noble, Dustin Scheinost.

# Model training

The code for training models and evaluating their within-dataset performance is found in the file [run_training.py](run_training.py). 

You can input the phenotype you want to predict, the training dataset, the random seed, and the number of training samples.

Example call:
```
python run_training.py --pheno age --train_dataset pnc --train_seed 0 --num_train 200
```

# External validation

The code for applying the above models to an *external* dataset is found in the file [run_external_validation.py](run_external_validation.py). 

You can input the phenotype you want to predict, the training dataset, the random seed, and the number of training samples. This code is set up to evaluate the model in all available test datasets for all relevant external sample sizes.

Example call:
```
python run_external_validation.py --pheno age --train_dataset pnc --train_seed 0 --num_train 200
```

# Reading/processing data

The data should be converted to a form more suitable for our plotting script using the notebook [read_process_results.ipynb](read_process_results.ipynb).

# Plotting

The notebook for plotting the results is [plots.ipynb](plots.ipynb). This notebook includes the calculation of power, effect size inflation, and the difference between internal/external performance. 


Please check back shortly for an interactive jupyter notebook (via google colab) through which you can plot our results
