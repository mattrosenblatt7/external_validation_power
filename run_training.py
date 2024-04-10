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

# Define several functions
def calc_q2(true_vals, pred_vals):
    # calculate q2 (cross-validated r2)
    return 1 - ((true_vals-pred_vals)**2).mean() / ((true_vals-true_vals.mean())**2).mean()
 
def r_to_p(r, n):
    # convert r values to p values: FOR FEATURE SELECTION (two tails)
    t = r / np.sqrt((1-r**2)/ (n-2) )
    p = 2*stats.t.sf(abs(t), df=n-2)
    return p

def my_custom_loss_func(y_true, y_pred):
    # custom scorer (Pearson's r) for grid search
    return np.corrcoef(y_true, y_pred)[0, 1]


# add any arguments
parser = argparse.ArgumentParser()
parser.add_argument("--pheno", type=str, help="which phenotype", choices=['age', 'ap', 'mr', 'mr_scaled', 'bmi', 'wm', 'wm_corrected', 'ad',
                                                                            'shape', 'rel', 'match'])  # which phenotype to train
parser.add_argument("--train_dataset", type=str, help="dataset to train in", choices=['abcd', 'hbn', 'hcpd', 'pnc',
                                                                                        'hcp', 'hcp_sc', 'chcp', 'chcp_sc',
                                                                                        'hbn_sc', 'hcpd_sc', 'qtab_sc',
                                                                                        'abcd_1_scans', 'abcd_2_scans', 'abcd_3_scans', 'abcd_4_scans'])  # name of training dataset
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

# can alternatively expand this, but for now just one training dataset at a time
dataset_names = [train_dataset]

# load in all data
if train_dataset=='hcp' or train_dataset=='hcp_sc' or train_dataset=='chcp' or train_dataset=='chcp_sc':  # adult datasets
    load_path = '/home/mjr239/project/repro_data/new_datasets'
    save_path = '/gpfs/gibbs/project/scheinost/mjr239/repro/updated_code_feb2024/results/adult/'
    df_dataset_pheno_sample_size = pd.read_csv(os.path.join(load_path, 'adult_pheno_dataset_sample_size.csv')) 
    run_perm = False  # whether to run permutation testing
elif train_dataset=='hbn_sc' or train_dataset=='hcpd_sc' or train_dataset=='qtab_sc':  # developmental structural connectivity datasets
    load_path = '/home/mjr239/project/repro_data/new_datasets'
    save_path = '/gpfs/gibbs/project/scheinost/mjr239/repro/updated_code_feb2024/results/dev_sc/'
    df_dataset_pheno_sample_size = pd.read_csv(os.path.join(load_path, 'dev_sc_pheno_dataset_sample_size.csv')) 
    run_perm = False  # whether to run permutation testing
elif train_dataset=='abcd_1_scans' or train_dataset=='abcd_2_scans' or train_dataset=='abcd_3_scans' or train_dataset=='abcd_4_scans':   # varying scan length datasets
    load_path = '/home/mjr239/project/repro_data/'
    save_path = '/gpfs/gibbs/project/scheinost/mjr239/repro/updated_code_feb2024/results/data_amount/'    
    df_dataset_pheno_sample_size = pd.read_csv(os.path.join(load_path, 'data_amount_pheno_dataset_sample_size.csv'))
    run_perm = False  # whether to run permutation testing
else:  # original datasets (main analysis)
    load_path = '/home/mjr239/project/repro_data/'
    save_path = '/gpfs/gibbs/project/scheinost/mjr239/repro/updated_code_feb2024/results/'
    df_dataset_pheno_sample_size = pd.read_csv(os.path.join(load_path, 'pheno_dataset_sample_size.csv'))
    run_perm = True  # whether to run permutation testing

datasets = dict()
for dname in dataset_names:
    
    # load in datasets
    try:
        dat = sio.loadmat( os.path.join(load_path, dname+'_feat.mat') )
    except NotImplementedError:  # if matlab file not compatible (v7.3), need to use hdf5 reader
        dat = mat73.loadmat(os.path.join(load_path, dname+'_feat.mat'))

    # read in phenotype and covariates
    df_tmp = pd.read_csv( os.path.join(load_path, dname+'_python_final.csv') )
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

    # get family structure
    if 'family_id' in all_keys:
        family_ids = np.array(datasets[dname]['behav']['family_id'])

# Train model
print('************Pheno {:s} seed {:d}*******************'.format(pheno, train_seed))

##### subsample the training dataset #####
np.random.seed(train_seed)
if 'family_id' in all_keys:  # avoid family leakage if family ID is a variable
    unique_family_ids = np.unique(family_ids)
    num_unique_families = len(unique_family_ids)
    shuffle_family_ids = unique_family_ids[ np.random.permutation( num_unique_families ) ] # shuffle family IDs
    members_per_family = datasets[train_dataset]['n'] / num_unique_families
    num_family_heldout = int(np.round(datasets[dname]['heldout_size'] / members_per_family ))
    num_family_train = int(np.round( num_train / members_per_family ))
    heldout_families = shuffle_family_ids[:num_family_heldout]
    train_families = shuffle_family_ids[num_family_heldout:(num_family_heldout+num_family_train)]
    heldout_idx = np.where([fid in heldout_families for fid in family_ids])[0]
    train_subset_idx = np.where([fid in train_families for fid in family_ids])[0]
    
else:  # no family ID variable
    shuffle_idx = np.random.permutation(datasets[train_dataset]['n'])  # shuffle all indices
    heldout_idx = shuffle_idx[:datasets[dname]['heldout_size']]  # heldout eval data
    train_subset_idx = shuffle_idx[datasets[dname]['heldout_size']:(datasets[dname]['heldout_size']+num_train)]  # training subset
    

# get heldout subset
X_heldout = datasets[train_dataset]['X'][:, heldout_idx].T
y_heldout = np.squeeze(np.array(datasets[train_dataset]['behav'][pheno])[heldout_idx])
C_heldout = datasets[train_dataset]['C_all'][heldout_idx, :]

# get training subset
X_train = datasets[train_dataset]['X'][:, train_subset_idx].T
y_train = np.squeeze(np.array(datasets[train_dataset]['behav'][pheno])[train_subset_idx])
C_train = datasets[train_dataset]['C_all'][train_subset_idx, :]  # covariates

# covariate regression
Beta = np.matmul( np.matmul( np.linalg.inv( np.matmul(C_train.T, C_train)), C_train.T), X_train)
X_train = X_train - np.matmul(C_train, Beta)
    
# feature selection
r = r_regression(X_train, y_train)
p = r_to_p(r, len(y_train)) 
pthresh = np.percentile(p, 100*per_feat)
sig_feat_loc = np.where(p<pthresh)[0]
    
# model fitting
inner_cv = KFold(n_splits=k, shuffle=True, random_state=train_seed)  # inner cv based on training seed
regr = GridSearchCV(estimator=Ridge(), param_grid={'alpha':np.logspace(-3, 3, 7)}, cv=inner_cv, scoring=score)
regr.fit(X_train[:, sig_feat_loc], y_train)

# prediction  in held-out subset
X_heldout = X_heldout - np.matmul(C_heldout, Beta)  # regress covariates from held-out data
yp_heldout = regr.predict(X_heldout[:, sig_feat_loc])

# evaluation in held-out subset
r_eval = np.corrcoef(yp_heldout, y_heldout)[0,1]
q2_eval = calc_q2(y_heldout, yp_heldout)


# get mean absolute error, and run permutation test for p values
mae = np.mean( np.abs( yp_heldout - y_heldout ) )
if run_perm:
    num_perm = 10000
    mae_null = np.zeros((num_perm,))
    r_null = np.zeros((num_perm,))
    n_eval = len(y_heldout)
    for perm_idx in range(num_perm):  # permutation test
        np.random.seed(seed=perm_idx)
        shuffle_idx = np.random.permutation(n_eval)  # shuffle indices
       
        # get null MAE
        mae_null[perm_idx] = np.mean( np.abs( y_heldout - yp_heldout[shuffle_idx] ) ) 
        
        # get null correlation
        r_null[perm_idx] = np.corrcoef(y_heldout, yp_heldout[shuffle_idx])[0, 1]
    # get p values
    p_perm_mae = np.mean(mae>mae_null)
    p_perm_r = np.mean(r_eval<r_null)
else:  # if not running permutation tests
    p_perm_mae = np.nan
    p_perm_r = np.nan

# save model and performance
save_name = pheno + '_train_' + train_dataset + '_trainsize_' + str(num_train) + '_seed_' + str(train_seed) + '.npz'
np.savez(os.path.join(save_path, 'repro_results_within', save_name),
        r=r_eval,
        q2=q2_eval,
        mae=mae,
        p_perm_mae=p_perm_mae,
        p_perm_r=p_perm_r,
         mask=sig_feat_loc,
         coef = regr.best_estimator_.coef_,
         intercept=regr.best_estimator_.intercept_,
         Beta_covar_feat = Beta[:, sig_feat_loc],
         heldout_size=datasets[dname]['heldout_size']
        )