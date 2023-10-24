'''
Code used for training models and evaluating their performance in a held-out within-dataset subset
'''

import pandas as pd
import numpy as np
import scipy.io as sio
import os
import time
import random
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.feature_selection import SelectPercentile, f_regression, r_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
import argparse
import mat73


def calc_q2(true_vals, pred_vals):
    # calculate q2 (cross-validated r2)
    return 1 - ((true_vals-pred_vals)**2).mean() / ((true_vals-true_vals.mean())**2).mean()
 
def r_to_p(r, n):
    # convert r values to p values
    t = r / np.sqrt((1-r**2)/ (n-2) )
    p = 2*stats.t.sf(abs(t), df=n-2)
    return p


def my_custom_loss_func(y_true, y_pred):
    # custom scorer (Pearson's r) for grid search
    return np.corrcoef(y_true, y_pred)[0, 1]


# add any arguments
parser = argparse.ArgumentParser()
parser.add_argument("--pheno", type=str, help="which phenotype", choices=['age', 'attn', 'mr', 'mr_scaled'])  # which phenotype to train
parser.add_argument("--train_dataset", type=str, help="dataset to train in", choices=['abcd', 'hbn', 'hcpd', 'pnc'])  # name of training dataset
parser.add_argument("--train_seed", type=int, help="dataset to train in", default=0)  # random seed for training dataset
parser.add_argument("--num_train", type=int, help="size of training dataset", default=400)  # number of training samples

# parse arguments
args = parser.parse_args()
pheno = args.pheno
train_dataset = args.train_dataset
num_train = args.num_train
train_seed = args.train_seed

# Set running parameters
per_feat = 0.01  # percentage of features for feature selection
k = 5  # number of folds for grid search
score = make_scorer(my_custom_loss_func, greater_is_better=True)

# can alteernatively expand this, but for now just one training dataset at a time
dataset_names = [train_dataset]

# load in all data
df_dataset_pheno_sample_size = pd.read_csv(os.path.join('/home/mjr239/project/repro_data/', 'pheno_dataset_sample_size.csv'))
datasets = dict()
for dname in dataset_names:
    
    # load in datasets
    try:
        dat = sio.loadmat( os.path.join('/home/mjr239/project/repro_data/', dname+'_feat.mat') )
    except NotImplementedError:  # if matlab file not compatible (v7.3), need to use hdf5 reader
        dat = mat73.loadmat(os.path.join('/home/mjr239/project/repro_data/', dname+'_feat.mat'))

    # read in phenotype and covariates
    df_tmp = pd.read_csv( os.path.join('/home/mjr239/project/repro_data/', dname+'_python_table.csv') )
    all_possible_covars = ['age', 'sex', 'motion_vals', 'site', 'family_id']
    df_include_vars = [c for c in all_possible_covars if ((c in df_tmp.keys()) & (pheno!=c))]  # covariate variable names (age excluded when age is the phenotype to predict)
    df_include_vars.append(pheno)
    df_tmp = df_tmp[df_include_vars]
    good_idx = np.where(df_tmp.isna().sum(axis=1)==0)[0]  # remove rows with missing data
    df_tmp = df_tmp.iloc[good_idx, :]
    df_tmp['sex'] = df_tmp.sex.replace('F', 0).replace('M', 1)  # replace self-reported sex variables so they are 0/1

    # add this to dictionary of datasets
    datasets[dname] = dict()
    datasets[dname]['X'] = dat['X'][:, good_idx]   
    datasets[dname]['behav'] = df_tmp    
    datasets[dname]['n'] = len(datasets[dname]['behav'])
    
    # add in covariate array to dictionary
    all_keys = datasets[dname]['behav'].keys()
    covar_keys = [k for k in all_keys if ((k!=pheno) and (k!='site') and (k!='family_id'))]  # motion, sex, age
    datasets[dname]['C_all'] = np.array(datasets[dname]['behav'][covar_keys])
    
    # get size of held-out data based on pre-set sizes
    datasets[dname]['heldout_size'] = int(df_dataset_pheno_sample_size[(df_dataset_pheno_sample_size.dataset==dname) & (df_dataset_pheno_sample_size.pheno==pheno)].n_heldout)

# Train model
datasets[train_dataset]['r'] = np.nan
datasets[train_dataset]['q2'] = np.nan

print('************Pheno {:s} seed {:d}*******************'.format(pheno, train_seed))

##### subsample the training dataset #####
np.random.seed(train_seed)
shuffle_idx = np.random.permutation(datasets[train_dataset]['n'])  # shuffle all indices

# get heldout subset
heldout_idx = shuffle_idx[:datasets[dname]['heldout_size']]
X_heldout = datasets[train_dataset]['X'][:, heldout_idx].T
y_heldout = np.squeeze(np.array(datasets[train_dataset]['behav'][pheno])[heldout_idx])
C_heldout = datasets[train_dataset]['C_all'][heldout_idx, :]

# get training subset
subset_idx = shuffle_idx[datasets[dname]['heldout_size']:(datasets[dname]['heldout_size']+num_train)]
X_train = datasets[train_dataset]['X'][:, subset_idx].T
y_train = np.squeeze(np.array(datasets[train_dataset]['behav'][pheno])[subset_idx])
C_train = datasets[train_dataset]['C_all'][subset_idx, :]  # covariates

# covariate regression
Beta = np.matmul( np.matmul( np.linalg.inv( np.matmul(C_train.T, C_train)), C_train.T), X_train)
X_train = X_train - np.matmul(C_train, Beta)
    
# feature selection
r = r_regression(X_train, y_train)
p = r_to_p(r, len(y_train)) 
pthresh = np.percentile(p, 100*per_feat)
sig_feat_loc = np.where(p<pthresh)[0]
    
# model fitting
# for kfold_seed_idx, kfold_seed in enumerate(range(num_kfold_repeats)):
kfold_seed = 0
inner_cv = KFold(n_splits=k, shuffle=True, random_state=kfold_seed) 
regr = GridSearchCV(estimator=Ridge(), param_grid={'alpha':np.logspace(-3, 3, 7)}, cv=inner_cv, scoring=score)
regr.fit(X_train[:, sig_feat_loc], y_train)

# prediction and evaluation in held-out subset
X_heldout = X_heldout - np.matmul(C_heldout, Beta)  # regress covariates from held-out data
yp_heldout = regr.predict(X_heldout[:, sig_feat_loc])

r_eval = np.corrcoef(yp_heldout, y_heldout)
q2_eval = calc_q2(y_heldout, yp_heldout)

# save model and performance
save_name = pheno + '_train_' + train_dataset + '_trainsize_' + str(num_train) + '_seed_' + str(train_seed) + '.npz'
np.savez(os.path.join('/gpfs/gibbs/project/scheinost/mjr239/repro/results/repro_results_within', save_name),
        r=r_eval[0, 1],
        q2=q2_eval,
         mask=sig_feat_loc,
         coef = regr.best_estimator_.coef_,
         intercept=regr.best_estimator_.intercept_,
         Beta_covar_feat = Beta[:, sig_feat_loc],
         heldout_size=datasets[dname]['heldout_size']
        )