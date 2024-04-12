This repository contains code corresponding to the manuscript ["Power and reproducibility in the external validation of brain-phenotype predictions"](https://www.biorxiv.org/content/10.1101/2023.10.25.563971v1) by Matthew Rosenblatt, Link Tejavibulya, Huili Sun, Chris C. Camp, Milana Khaitova, Brendan D. Adkinson, Rongtao Jiang, Margaret L. Westwater, Stephanie Noble, Dustin Scheinost.

# Software requirements

This code requires Python 3. While it may work with various versions of Python and the following packages, the code was specifically developed and tested with Python 3.11.3 and the following packages:

* mat73 0.60
* matplotlib 3.8.0
* numpy 1.24.3
* pandas 2.0.3
* scikit-learn 1.2.2
* scipy 1.10.1
* seaborn 0.13.0
* plotly 5.19.0
* tqdm 4.66.1

Beyond installing python and these packages, no specific installation is required. Installation of python and the packages should take about 10 minutes. To reproduce the plots, run [make_plots.ipynb](./make_plots.ipynb).

# Model training

The code for training models and evaluating their within-dataset performance is found in the file [run_training.py](run_training.py). 

You can input the phenotype you want to predict, the training dataset, the random seed, and the number of training samples.

Example call:
```
python run_training.py --pheno age --train_dataset pnc --train_seed 0 --num_train 200
```

The code will save files with the model coefficients and internal validation performance (please see the relevant paths in the script to edit the saved output).

Please note that the analysis in this paper is likely not practical to run on a personal computer, due to the many simulations and computational resources involved.

# External validation

The code for applying the above models to an *external* dataset is found in the file [run_external_validation.py](run_external_validation.py). 

You can input the phenotype you want to predict, the training dataset, the random seed, and the number of training samples. This code is set up to evaluate the model in all available test datasets for all relevant external sample sizes.

Example call:
```
python run_external_validation.py --pheno age --train_dataset pnc --train_seed 0 --num_train 200
```
The code will save files with the external validation performance, as well as the corresponding internal validation performance (please see the relevant paths in the script to edit the saved output).

Please note that the analysis in this paper is likely not practical to run on a personal computer, due to the many simulations and computational resources involved.

# Reading/processing data

The data should be converted to a form more suitable for our plotting script using the script [read_results.py](./read_results.py).

Then, these results are aggregated in [process_csv_files.ipynb](./process_csv_files.ipynb).

# Plotting

The notebook for plotting the results is [make_plots.ipynb](./make_plots.ipynb).

# License

This code is covered under the MIT License.
