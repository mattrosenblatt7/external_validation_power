'''
Evaluate models in external datasets
'''
import pandas as pd
import numpy as np
import scipy.io as sio
import os
import time
import random
# from tqdm import tqdm
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
parser.add_argument("--pheno", type=str, help="which phenotype", choices=['age', 'attn', 'mr', 'mr_scaled'])  # which phenotype to predict
parser.add_argument("--train_dataset", type=str, help="dataset to train in", choices=['abcd', 'hbn', 'hcpd', 'pnc'])  # name of training dataset
parser.add_argument("--train_seed", type=int, help="dataset to train in", default=0)  # random seed for training dataset
parser.add_argument("--num_train", type=int, help="size of training dataset", default=400)  # number of training samples

# parse arguments
args = parser.parse_args()
pheno = args.pheno
train_dataset = args.train_dataset
train_seed = args.train_seed
num_train = args.num_train

# resample external dataset 100x
num_test_seeds = 100

# set save and load paths
data_load_path = '/home/mjr239/project/repro_data/'  # /home/mjr239/project/repro_data/
model_load_path = '/gpfs/gibbs/project/scheinost/mjr239/repro/results/repro_results_within'  # /gpfs/gibbs/project/scheinost/mjr239/repro/results/repro_results_within
save_path = '/gpfs/gibbs/project/scheinost/mjr239/repro/results/repro_results_across'

# load in results of trained model
model_load_name = pheno + '_train_' + train_dataset + '_trainsize_' + str(num_train) + '_seed_' + str(train_seed) + '.npz'
model_and_results = np.load(os.path.join(model_load_path, model_load_name))

# load in external dataset
df_dataset_pheno_sample_size = pd.read_csv(os.path.join(data_load_path, 'pheno_dataset_sample_size.csv'))

# get all test datasets
if pheno=='age':  # no age in abcd
    all_datasets = ['hbn', 'hcpd', 'pnc']
elif pheno=='mr_scaled':  # none for pnc
    all_datasets = ['abcd', 'hbn', 'hcpd']
else:  
    all_datasets = ['abcd', 'hbn', 'hcpd', 'pnc']
test_datasets = [d for d in all_datasets if d!=train_dataset]

# loop over all test/external datasets
for test_dataset in test_datasets:
    
    # load in datasets
    datasets = dict()
    try:
        dat = sio.loadmat( os.path.join(data_load_path, test_dataset+'_feat.mat') )
    except NotImplementedError:  # if matlab file not compatible (v7.3), need to use hdf5 reader
        dat = mat73.loadmat(os.path.join(data_load_path, test_dataset+'_feat.mat'))
    
    # read in phenotype and covariates
    df_tmp = pd.read_csv( os.path.join(data_load_path, test_dataset+'_python_table.csv') )
    all_possible_covars = ['age', 'sex', 'motion_vals', 'site', 'family_id']
    df_include_vars = [c for c in all_possible_covars if ((c in df_tmp.keys()) & (pheno!=c))]  # covariate variable names (age excluded when age is the phenotype to predict)
    df_include_vars.append(pheno)
    df_tmp = df_tmp[df_include_vars]
    good_idx = np.where(df_tmp.isna().sum(axis=1)==0)[0]  # remove rows with missing data
    df_tmp = df_tmp.iloc[good_idx, :]
    df_tmp['sex'] = df_tmp.sex.replace('F', 0).replace('M', 1)  # replace self-reported sex variables so they are 0/1
    
    # add this to dictionary of datasets
    datasets[test_dataset] = dict()
    datasets[test_dataset]['X'] = dat['X'][:, good_idx]   
    datasets[test_dataset]['behav'] = df_tmp    
    datasets[test_dataset]['n'] = len(datasets[test_dataset]['behav'])
    
    # add in covariate array to dictionary
    all_keys = datasets[test_dataset]['behav'].keys()
    covar_keys = [k for k in all_keys if ((k!=pheno) and (k!='site') and (k!='family_id'))]  # motion, sex, age
    datasets[test_dataset]['C_all'] = np.array(datasets[test_dataset]['behav'][covar_keys])
    
    # get all possible sample sizes in the test dataset
    # 20 points logspace from n=20 to n=max across all datasets/pheno
    possible_n = np.round(np.logspace(np.log10(20), np.log10(df_dataset_pheno_sample_size.n_train.max()), 20))  # np.round(np.logspace(np.log10(20), np.log10(4149), 12)) 
    possible_n = [int(p) for p in possible_n]
    n_max_test = int(df_dataset_pheno_sample_size.loc[(df_dataset_pheno_sample_size.dataset==test_dataset) & (df_dataset_pheno_sample_size.pheno==pheno), 'n'])
    n_all_test = [p for p in possible_n if p<n_max_test] + [n_max_test]
    
    
    # evaluate in all data (not yet resampled) using saved coefficients for model/covariates
    X_all_eval = np.copy(datasets[test_dataset]['X'].T)
    y_all_eval = np.squeeze(np.array(datasets[test_dataset]['behav'][pheno]))
    C_all_eval = datasets[test_dataset]['C_all']
    X_all_eval_corrected = X_all_eval[:, model_and_results['mask']] - np.matmul(C_all_eval, model_and_results['Beta_covar_feat'])
    yp_all_eval = np.ravel(np.matmul(X_all_eval_corrected, model_and_results['coef'][:, None] )
                           + model_and_results['intercept'] )
    
    # loop over all test sample sizes
    num_test_list = []
    test_seed_list = []
    r_external_list = []
    q2_external_list = []
    for num_test in n_all_test:
        
        # for each sample size, loop over seeds
        r_eval = np.zeros((num_test_seeds, ))  # initialize for all seeds
        q2_eval = np.zeros((num_test_seeds, ))
        for test_seed in range(num_test_seeds):
            
            # set seed and shuffle indices for resampling
            np.random.seed(test_seed)
            shuffle_idx = np.random.permutation(datasets[test_dataset]['n'])  # shuffle all indices

            # get subset for evaluation
            eval_idx = shuffle_idx[:num_test]

            # evaluate performance in subsampled data
            r_eval[test_seed] = np.corrcoef(y_all_eval[eval_idx], yp_all_eval[eval_idx])[0, 1]
            q2_eval[test_seed] = calc_q2(y_all_eval[eval_idx], yp_all_eval[eval_idx])
            
            num_test_list.append(num_test)
            test_seed_list.append(test_seed)
            r_external_list.append(r_eval[test_seed])
            q2_external_list.append(q2_eval[test_seed])
    
    # store in lists to eventually save
    train_dataset_list = len(num_test_list)*[train_dataset]
    test_dataset_list = len(num_test_list)*[test_dataset]
    num_train_list = len(num_test_list)*[num_train]
    train_seed_list = len(num_test_list)*[train_seed]
    pheno_list = len(num_test_list)*[pheno]
    r_internal_list = len(num_test_list)*[model_and_results['r']]
    q2_internal_list = len(num_test_list)*[model_and_results['q2']]

    
    # save results
    results_save_name = 'results_train_' + train_dataset + '_size_' + str(num_train) + '_seed_' + str(train_seed) + '_test_' + test_dataset + '_pheno_' + pheno + '.npz'
    np.savez(os.path.join(save_path, results_save_name), train_dataset_list=train_dataset_list,
             pheno_list=pheno_list, test_dataset_list=test_dataset_list,
             train_seed_list=train_seed_list, num_train_list=num_train_list,
             test_seed_list=test_seed_list, num_test_list=num_test_list,
             r_internal_list=r_internal_list, q2_internal_list=q2_internal_list,
             r_external_list=r_external_list, q2_external_list=q2_external_list)
