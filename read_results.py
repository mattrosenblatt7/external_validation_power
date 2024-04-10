'''
Code to aggregate and save results as .npz files
'''

import numpy as np
import pandas as pd
import os
import glob
import re
import time
from tqdm.auto import tqdm

base_path = '/home/mjr239/project/repro/updated_code_feb2024/results'

################ EXTERNAL VALIDATION RESULTS################
'''
All variable names
train_dataset_list=train_dataset_list, pheno_list=pheno_list, test_dataset_list=test_dataset_list,
train_seed_list=train_seed_list, num_train_list=num_train_list,
test_seed_list=test_seed_list, num_test_list=num_test_list,
r_internal_list=r_internal_list, q2_internal_list=q2_internal_list, mae_internal_list=mae_internal_list,
p_perm_r_internal_list=p_perm_r_internal_list, p_perm_mae_internal_list=p_perm_mae_internal_list,
r_external_list=r_external_list, q2_external_list=q2_external_list, mae_external_list=mae_external_list,
p_perm_r_external_list=p_perm_r_external_list, p_perm_q2_external_list=p_perm_q2_external_list, p_perm_mae_external_list=p_perm_mae_external_list
'''

# get phenotype list
df_sample_size = pd.read_csv('/home/mjr239/project/repro_data/pheno_dataset_sample_size.csv')
pheno_all = list(df_sample_size.pheno.unique())
                                                                                                                                       

# loop over all phenotypes
for pheno_name in pheno_all:

    # get relevant datasets for each phenotype
    if pheno_name=='mr_scaled':  # no pnc in scaled matrix reasoning
        all_datasets = ['abcd', 'hbn', 'hcpd']
    elif pheno_name=='wm_corrected':  # no pnc in scaled working memory
        all_datasets = ['abcd', 'hbn', 'hcpd']
    else:
        all_datasets = ['abcd', 'hbn', 'hcpd', 'pnc']
    
    # loop over training datasets
    for train_dname in all_datasets:  
    
        # get corresponding test datasets
        test_datasets = [dname for dname in all_datasets if dname!=train_dname]
        
        # loop over all test datasets
        for test_dname in test_datasets:  
            
            # set savename and check if file exists
            save_name = os.path.join(base_path, 'across_aggregated', 'results_train_' + train_dname + '_test_' + test_dname + '_pheno_' + pheno_name + '.npz')
            if os.path.isfile(save_name):
                print('Skipping for training in {:s} and testing in {:s} for {:s}'.format(train_dname,
                                                                                            test_dname,
                                                                                            pheno_name))
                continue  # skipp this loop iteration if file exists
            
            # get files for this train/test dataset combination
            file_list = glob.glob(os.path.join(base_path, 'repro_results_across/results_train_' + train_dname + '*_test_' + test_dname  +'_pheno_' + pheno_name + '.npz') )

        
            print('********Number of external validation result files for training in {:s} and testing in {:s} for {:s}: {:d}********'.format( train_dname,
                                                                                                                                        test_dname,
                                                                                                                                        pheno_name,
                                                                                                                                        len(file_list) ) )
            # reset/initialize lists to store data
            train_dataset = []
            pheno = []
            test_dataset = []
            train_seed = []
            num_train = []
            test_seed = []
            num_test = []
            r_internal = []
            q2_internal = []
            mae_internal = []
            p_perm_r_internal = []
            p_perm_mae_internal = []
            r_external = []
            q2_external = []
            mae_external = []
            p_perm_r_external = []
            p_perm_q2_external = []
            p_perm_mae_external = []
            
            # loop over all these files, saving them as .npz files
            t1 = time.time()  # start time
            for fname in tqdm(file_list):
            
                # load data
                dat = np.load(fname)
            
                # put data in lists
                train_dataset.extend(dat['train_dataset_list'])
                pheno.extend(dat['pheno_list'])
                test_dataset.extend(dat['test_dataset_list'])
                train_seed.extend(dat['train_seed_list'])
                num_train.extend(dat['num_train_list'])
                test_seed.extend(dat['test_seed_list'])
                num_test.extend(dat['num_test_list'])
                r_internal.extend(dat['r_internal_list'])
                q2_internal.extend(dat['q2_internal_list'])
                mae_internal.extend(dat['mae_internal_list'])
                p_perm_r_internal.extend(dat['p_perm_r_internal_list'])
                p_perm_mae_internal.extend(dat['p_perm_mae_internal_list'])
                r_external.extend(dat['r_external_list'])
                q2_external.extend(dat['q2_external_list'])
                mae_external.extend(dat['mae_external_list'])
                p_perm_r_external.extend(dat['p_perm_r_external_list'])
                p_perm_q2_external.extend(dat['p_perm_q2_external_list'])
                p_perm_mae_external.extend(dat['p_perm_mae_external_list'])
            
                
            print('Elapsed time: {:.2f}'.format(time.time()-t1))
            # save results
            np.savez(save_name,
                    train_dataset=train_dataset, pheno=pheno, test_dataset=test_dataset, 
                    train_seed=train_seed, num_train=num_train, 
                    test_seed=test_seed, num_test=num_test,
                    r_internal=r_internal, q2_internal=q2_internal, mae_internal=mae_internal,
                    p_perm_r_internal=p_perm_r_internal, p_perm_mae_internal=p_perm_mae_internal,
                    r_external=r_external, q2_external=q2_external, mae_external=mae_external,
                    p_perm_r_external=p_perm_r_external, p_perm_q2_external=p_perm_q2_external, p_perm_mae_external=p_perm_mae_external
                    )
                

################ INTERNAL VALIDATION RESULTS################

'''
All variable names:
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
'''

# loop over training datasets
for train_dname in ['abcd', 'hbn', 'hcpd', 'pnc']:
    # get list of all results files
    file_list = glob.glob(os.path.join(base_path, 'repro_results_within/*_train_' + train_dname + '*.npz') )
    
    print('********Number of internal validation result files to load for training {:s}: {:d}********'.format( train_dname, len(file_list) ) )
    
    # initialize empty lists
    r = []
    q2 = []
    mae = []
    p_perm_mae = []
    p_perm_r = []
    pheno = []
    dataset = []
    sample_size = []
    resample_seed = []
    kfold_seed = []
    num_kfold_repeats = 10
    
    # loop over all files, save 
    for idx, fname in enumerate(file_list):
        
        if idx%1000==0:
            print('Within progress: {:d}/{:d}'.format(idx, len(file_list) ) )
        
        # load saved data
        dat = np.load(fname)
        
        # extend lists
        r.extend(list(dat['r']))
        q2.extend(list(dat['q2']))
        mae.extend(list(dat['mae']))
        p_perm_mae.extend(list(dat['p_perm_mae']))
        p_perm_r.extend(list(dat['p_perm_r']))
        
        pheno_tmp = num_kfold_repeats*[re.search('/repro_results_within/(.*)_train_', fname).group(1)]
        pheno.extend(pheno_tmp)
        
        dataset_tmp = num_kfold_repeats*[re.search('_train_(.*)_trainsize', fname).group(1)]
        dataset.extend(dataset_tmp)
        
        sample_size_tmp = num_kfold_repeats*[ int(re.search('_trainsize_(.*)_seed_', fname).group(1)) ]
        sample_size.extend(sample_size_tmp)
        
        resample_seed_tmp = num_kfold_repeats*[ int(re.search('_seed_(.*).npz', fname).group(1)) ]
        resample_seed.extend(resample_seed_tmp)
        
        kfold_seed.extend(list(range(num_kfold_repeats)))
    
    # make and save dataframe
    df_results = pd.DataFrame({'pheno':pheno, 'dataset':dataset,
                               'sample_size':sample_size, 'resample_seed':resample_seed,
                               'kfold_seed':kfold_seed, 
                               'r':r, 'q2':q2, 'mae':mae,
                                'p_perm_r':p_perm_r, 'p_perm_mae':p_perm_mae
    })
    df_results.to_csv(os.path.join(base_path, 'processed_results', 'repro_within_' + train_dname + '.csv', index=False) )
