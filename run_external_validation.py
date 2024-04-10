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
from tqdm.auto import tqdm

# Define several functions
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
    
def corr2_coeff(A, B):
    '''
    Calculate correlation coefficient between array and vector
    From https://stackoverflow.com/questions/30143417/computing-the-correlation-coefficient-between-two-multi-dimensional-arrays/30143754#30143754
    '''
    
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))
    
def get_p_vals(y_true, y_predicted, r, q2, mae, num_perm=500):
    '''
    Function to get null distributions and p values
    This update uses matrices to make the computations much faster
    '''
    # get shuffled indices: num_perm rows x num_participants columns
    shuffle_idx_all = np.array([ np.random.permutation(len(y_true)) for p in range(num_perm) ]  )
       
    # calculate null distributions
    mae_null = np.abs(y_true - y_predicted[shuffle_idx_all]).mean(axis=1)
    r_null = corr2_coeff(y_true.reshape(len(y_true),1).T, y_predicted[shuffle_idx_all]).ravel()
    q2_null = 1-((y_true-y_predicted[shuffle_idx_all])**2).mean(axis=1) / ((y_true-y_true.mean())**2).mean()

    # get p values
    p_perm_r = np.mean(r<r_null)
    p_perm_q2 = np.mean(q2<q2_null)
    p_perm_mae = np.mean(mae>mae_null)
    
    return p_perm_r, p_perm_q2, p_perm_mae


# add any arguments
parser = argparse.ArgumentParser()
parser.add_argument("--pheno", type=str, help="which phenotype", choices=['age', 'ap', 'mr', 'mr_scaled', 'bmi', 'wm', 'wm_corrected', 'ad',
                                                                            'shape', 'rel', 'match'])  # which phenotype to predict
parser.add_argument("--train_dataset", type=str, help="dataset to train in", choices=['abcd', 'hbn', 'hcpd', 'pnc',
                                                                                        'hcp', 'hcp_sc', 'chcp', 'chcp_sc',
                                                                                        'hbn_sc', 'hcpd_sc', 'qtab_sc',
                                                                                        'abcd_1_scans', 'abcd_2_scans', 'abcd_3_scans', 'abcd_4_scans'])  # name of training dataset
parser.add_argument("--train_seed", type=int, help="dataset to train in", default=0)  # random seed for training dataset
parser.add_argument("--num_train", type=int, help="size of training dataset", default=400)  # number of training samples
parser.add_argument("--data_amount_analysis", type=int, help="whether this analysis is varying scan length", default=0)  # whether scan length is varied

# parse arguments
args = parser.parse_args()
pheno = args.pheno
train_dataset = args.train_dataset
train_seed = args.train_seed
num_train = args.num_train
data_amount_analysis = args.data_amount_analysis

# resample external dataset 100x
num_test_seeds = 100

# set save and load paths
if train_dataset=='hcp' or train_dataset=='hcp_sc' or train_dataset=='chcp' or train_dataset=='chcp_sc':  # adult datasets
    
    # set load/save paths
    data_load_path = '/home/mjr239/project/repro_data/new_datasets'  # /home/mjr239/project/repro_data/
    model_load_path = '/gpfs/gibbs/project/scheinost/mjr239/repro/updated_code_feb2024/results/adult/repro_results_within'  # /gpfs/gibbs/project/scheinost/mjr239/repro/results/repro_results_within
    save_path = '/gpfs/gibbs/project/scheinost/mjr239/repro/updated_code_feb2024/results/adult/repro_results_across'
    
    # load in sample size info
    df_dataset_pheno_sample_size = pd.read_csv(os.path.join(data_load_path, 'adult_pheno_dataset_sample_size.csv'))  # load in external dataset information
    
    # get all test datasets
    if (train_dataset=='hcp') or (train_dataset=='chcp'):  # adult functional connectivity
        all_datasets = ['hcp', 'chcp']
    elif (train_dataset=='hcp_sc') or (train_dataset=='chcp_sc'):  # adult structural connectivity
        all_datasets = ['hcp_sc', 'chcp_sc']
    test_datasets = [d for d in all_datasets if d!=train_dataset]
    
    run_perm = False  # whether to run permutation testing
elif train_dataset=='hbn_sc' or train_dataset=='hcpd_sc' or train_dataset=='qtab_sc':  # developmental structural connectome datasets
    
    # set load/save paths
    data_load_path = '/home/mjr239/project/repro_data/new_datasets'  # /home/mjr239/project/repro_data/
    model_load_path = '/gpfs/gibbs/project/scheinost/mjr239/repro/updated_code_feb2024/results/dev_sc/repro_results_within' 
    save_path = '/gpfs/gibbs/project/scheinost/mjr239/repro/updated_code_feb2024/results/dev_sc/repro_results_across'
    
    # load in sample size info
    df_dataset_pheno_sample_size = pd.read_csv(os.path.join(data_load_path, 'dev_sc_pheno_dataset_sample_size.csv')) 
    
    # get all test datasets
    all_datasets = ['hbn_sc', 'hcpd_sc', 'qtab_sc']
    test_datasets = [d for d in all_datasets if d!=train_dataset]
    
    run_perm = False  # whether to run permutation testing
elif train_dataset=='abcd_1_scans' or train_dataset=='abcd_2_scans' or train_dataset=='abcd_3_scans' or train_dataset=='abcd_4_scans':  # scan length (training) analysis
    
    # set load/save paths
    data_load_path = '/home/mjr239/project/repro_data/'
    model_load_path = '/gpfs/gibbs/project/scheinost/mjr239/repro/updated_code_feb2024/results/data_amount/repro_results_within' 
    save_path = '/gpfs/gibbs/project/scheinost/mjr239/repro/updated_code_feb2024/results/data_amount/repro_results_across'    

    # load in sample size info
    # need to combine dataframes so it has info from main datasets and scan length dataset
    df_dataset_pheno_sample_size_tmp = pd.read_csv(os.path.join(data_load_path, 'data_amount_pheno_dataset_sample_size.csv'))
    df_dataset_pheno_sample_size_main = pd.read_csv(os.path.join('/home/mjr239/project/repro_data/', 'pheno_dataset_sample_size.csv'))
    df_dataset_pheno_sample_size = pd.concat([df_dataset_pheno_sample_size_tmp, df_dataset_pheno_sample_size_main]).reset_index(drop=True)

    # get test datasets
    test_datasets = ['hbn', 'hcpd', 'pnc']
    
    run_perm = False  # whether to run permutation testing
elif (train_dataset=='hbn' or train_dataset=='hcpd' or train_dataset=='pnc') and (data_amount_analysis==1):  # scan length (testing) analysis
    
    # set load/save paths
    data_load_path = '/home/mjr239/project/repro_data/'  # /home/mjr239/project/repro_data/
    model_load_path = '/gpfs/gibbs/project/scheinost/mjr239/repro/updated_code_feb2024/results/repro_results_within'  # /gpfs/gibbs/project/scheinost/mjr239/repro/results/repro_results_within
    save_path = '/gpfs/gibbs/project/scheinost/mjr239/repro/updated_code_feb2024/results/data_amount/repro_results_across'    
    
    # load in sample size info
    # need to combine dataframes so it has info from main datasets and scan length dataset
    df_dataset_pheno_sample_size_tmp = pd.read_csv(os.path.join(data_load_path, 'data_amount_pheno_dataset_sample_size.csv'))
    df_dataset_pheno_sample_size_main = pd.read_csv(os.path.join('/home/mjr239/project/repro_data/', 'pheno_dataset_sample_size.csv'))
    df_dataset_pheno_sample_size = pd.concat([df_dataset_pheno_sample_size_tmp, df_dataset_pheno_sample_size_main]).reset_index(drop=True)
    
    # get all test datasets
    test_datasets = ['abcd_1_scans', 'abcd_2_scans', 'abcd_3_scans', 'abcd_4_scans']
    
    run_perm = False  # whether to run permutation testing
else:  # primary analysis
    
    # set load/save paths
    data_load_path = '/home/mjr239/project/repro_data/'  # /home/mjr239/project/repro_data/
    model_load_path = '/gpfs/gibbs/project/scheinost/mjr239/repro/updated_code_feb2024/results/repro_results_within'  # /gpfs/gibbs/project/scheinost/mjr239/repro/results/repro_results_within
    save_path = '/gpfs/gibbs/project/scheinost/mjr239/repro/updated_code_feb2024/results/repro_results_across'
    df_dataset_pheno_sample_size = pd.read_csv(os.path.join(data_load_path, 'pheno_dataset_sample_size.csv'))  # load in external dataset information
    
    # get all test datasets
    if (pheno=='mr_scaled') or (pheno=='wm_corrected'):  # scaled measures do not exist for pnc
        all_datasets = ['abcd', 'hbn', 'hcpd']
    else:  
        all_datasets = ['abcd', 'hbn', 'hcpd', 'pnc']
    test_datasets = [d for d in all_datasets if d!=train_dataset]
    
    run_perm = True  # whether to run permutation testing

# load in main dataset/phenotype .csv to get all possible sample sizes
df_dataset_pheno_sample_size_main = pd.read_csv(os.path.join('/home/mjr239/project/repro_data/', 'pheno_dataset_sample_size.csv'))

# load in results of trained model
model_load_name = pheno + '_train_' + train_dataset + '_trainsize_' + str(num_train) + '_seed_' + str(train_seed) + '.npz'
model_and_results = np.load(os.path.join(model_load_path, model_load_name))

# loop over all test/external datasets
for test_dataset in test_datasets:
    
    # load in datasets
    datasets = dict()
    try:
        dat = sio.loadmat( os.path.join(data_load_path, test_dataset+'_feat.mat') )
    except NotImplementedError:  # if matlab file not compatible (v7.3), need to use hdf5 reader
        dat = mat73.loadmat(os.path.join(data_load_path, test_dataset+'_feat.mat'))
    
    # read in phenotype and covariates
    df_tmp = pd.read_csv( os.path.join(data_load_path, test_dataset+'_python_final.csv') )
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
    # set number of different sample sizes
    num_points = 25
    possible_n = np.round(np.logspace(np.log10(20), np.log10(df_dataset_pheno_sample_size_main.n_train.max()), num_points))
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
    mae_external_list = []
    p_perm_r_external_list = []
    p_perm_q2_external_list = []
    p_perm_mae_external_list = []
    for num_test in tqdm(n_all_test):
        
        # for each sample size, loop over seeds
        for test_seed in range(num_test_seeds):
            
            # set seed and shuffle indices for resampling
            np.random.seed(test_seed)
            shuffle_idx = np.random.permutation(datasets[test_dataset]['n'])  # shuffle all indices

            # get subset for evaluation
            eval_idx = shuffle_idx[:num_test]

            # evaluate performance (r, q2, mae) in subsampled data
            r_eval = np.corrcoef(y_all_eval[eval_idx], yp_all_eval[eval_idx])[0, 1]
            q2_eval= calc_q2(y_all_eval[eval_idx], yp_all_eval[eval_idx])
            mae_eval = np.mean( np.abs( y_all_eval[eval_idx] - yp_all_eval[eval_idx] ) )
            
            #  get p values for r, q2, MAE with permutation tests
            if run_perm:
                p_perm_r, p_perm_q2, p_perm_mae = get_p_vals(y_all_eval[eval_idx], yp_all_eval[eval_idx],
                                                            r_eval, q2_eval, mae_eval,
                                                            num_perm=500)
            else:
                p_perm_r = np.nan
                p_perm_q2 = np.nan
                p_perm_mae = np.nan

            # make list of results
            num_test_list.append(num_test)
            test_seed_list.append(test_seed)
            r_external_list.append(r_eval)
            q2_external_list.append(q2_eval)
            mae_external_list.append(mae_eval)
            p_perm_r_external_list.append(p_perm_r)
            p_perm_q2_external_list.append(p_perm_q2)
            p_perm_mae_external_list.append(p_perm_mae)
    
    # store in lists to eventually save
    train_dataset_list = len(num_test_list)*[train_dataset]
    test_dataset_list = len(num_test_list)*[test_dataset]
    num_train_list = len(num_test_list)*[num_train]
    train_seed_list = len(num_test_list)*[train_seed]
    pheno_list = len(num_test_list)*[pheno]
    r_internal_list = len(num_test_list)*[model_and_results['r']]
    q2_internal_list = len(num_test_list)*[model_and_results['q2']]
    mae_internal_list = len(num_test_list)*[model_and_results['mae']]
    p_perm_r_internal_list = len(num_test_list)*[model_and_results['p_perm_r']]
    p_perm_mae_internal_list = len(num_test_list)*[model_and_results['p_perm_mae']]

    
    # save results
    results_save_name = 'results_train_' + train_dataset + '_size_' + str(num_train) + '_seed_' + str(train_seed) + '_test_' + test_dataset + '_pheno_' + pheno + '.npz'
    print('saving ' + results_save_name)
    np.savez(os.path.join(save_path, results_save_name), train_dataset_list=train_dataset_list,
             pheno_list=pheno_list, test_dataset_list=test_dataset_list,
             train_seed_list=train_seed_list, num_train_list=num_train_list,
             test_seed_list=test_seed_list, num_test_list=num_test_list,
             r_internal_list=r_internal_list, q2_internal_list=q2_internal_list, mae_internal_list=mae_internal_list,
             p_perm_r_internal_list=p_perm_r_internal_list, p_perm_mae_internal_list=p_perm_mae_internal_list,
             r_external_list=r_external_list, q2_external_list=q2_external_list, mae_external_list=mae_external_list,
             p_perm_r_external_list=p_perm_r_external_list, p_perm_q2_external_list=p_perm_q2_external_list, p_perm_mae_external_list=p_perm_mae_external_list)
