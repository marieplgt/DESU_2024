
# %% Imports

import copy
import os
import os.path as op
import shutil
from soundsemtools import utils as sstu
import soundsemtools as sst
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py

# import mne
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, KFold

def dircreate(dirs_to_create):
    """
    Create directories if they don't exist.

    :param dirs_to_create: Directory, or list of directories to create
    :type dirs_to_create: strings of path
    :return: Nothing, create the directories in the filesystem

    """
    if not op.exists(dirs_to_create):
        os.makedirs(dirs_to_create)

# # not present in my clone repo ph d
# dat_dir='/media/sf_sharedVM/github_repo/Aud2Sem_MEG/'
# tdistdir=dat_dir+'Toolboxes/DFI_iEEG/'
# import sys
# sys.path.append(tdistdir)
# # from tDistFuns import *


# %% Run the file as script

if __name__ == "__main__":

    username = "marieplgt"   # whoami within a terminal

    if username == "marieplgt":
        paths = sst.utils.get_proj_paths(
            "/media/sf_sharedVM/github_repo/phd/code/meg_analysis_code/config.json"
        )

    user = paths["user"]
    proj_dir = paths["proj_dir"]
    code_dir = paths["code_dir"]
    data_dir = paths["data_dir"]

    bids_dir = f"{proj_dir}/AUD2SEM_BIDS"


    my_figs_dir=bids_dir+'/derivatives/tmp_figures/'
    fig_path = f'{my_figs_dir}/linear_reg/'
    dircreate(fig_path)
    

    
    # # -------------------- Load W2V Distances --------------------------------------------------------
    # w2v_dir = '/media/sf_sharedVM/github_repo/Aud2Sem_MEG/code/meg_analysis_code/cf_pipelines/pipelines/derivatives/models/semantic/'
    # common_w2v_file = os.path.join(w2v_dir, 'common_w2v_concatenate_all_rdms_cosine.hdf5')

    # tmp = sst.rdm.load_rdms_h5py(common_w2v_file)
    # print(tmp.keys())

    # w2v_rdm = tmp['rdms_x']      # vector form : (n_pairs, n_features, n_rdms)
    # print(w2v_rdm.shape)

    # w2v_rdm_vec = w2v_rdm[:,:,0]
    # print(w2v_rdm_vec.shape)
    

    # # -------------------- Load Brain Distances --------------------------------------------------------

    # # Based on averaged epochs across 8 subjects
            
    # brain_rdms_crosslation_dir = '/media/marieplgt/T7/docs/PHD/Aud2sem_project/AUD2SEM_BIDS/derivatives/brain_rdms'
    # common_w2v_crosslation_file = os.path.join(brain_rdms_crosslation_dir, "sub-01_03_04_05_06_07_08_09_13_ch-meg_stim-comm_filt-LP_70_Hz_movavg-0_1_resamp-200_Hz_rdms-crosslation_cov-mne_empirical_cvscheme-cvloo.hdf5")
    
    # brain_rdms_crossnobis_dir = '/media/marieplgt/T7/docs/PHD/Aud2sem_project/AUD2SEM_BIDS/derivatives/brain_rdms'
    # common_w2v_crossnobis_file = os.path.join(brain_rdms_crossnobis_dir, "sub-01_03_04_05_06_07_08_09_13_ch-meg_stim-comm_filt-LP_70_Hz_movavg-0_1_resamp-200_Hz_rdms-crossnobis_cov-mne_empirical_cvscheme-cvloo.hdf5")

    # # crosslation distances
    # tmp_1 = sst.rdm.load_rdms_h5py(common_w2v_crosslation_file)     # LOAD AS RDM obj
    # # print(tmp_1.keys())
    # crosslation_rdm = tmp_1['rdms_x']      # vector form : (n_pairs, n_features, n_rdms)
    # # print(crosslation_rdm.shape)

    # # crossnobis distances
    # tmp_2 = sst.rdm.load_rdms_h5py(common_w2v_crossnobis_file)
    # # print(tmp_2.keys())
    # crossnobis_rdm = tmp_2['rdms_x']      # vector form : (n_pairs, n_features, n_rdms)
    # # print(crossnobis_rdm.shape)



    x=np.random.normal(0,1, 10000)
    y=np.random.normal(0,1, 10000)
    y=np.tile(y[...,None,None], 20)
    w2v_rdm_vec = x
    print(y.shape)

    times = np.arange(20)


    dist_metric = {'crosslation' : y,
                    'crossnobis': y,
                    }

    # print(dist_metric.keys())
    for metric in dist_metric.keys():
        rdm = dist_metric[metric]
        print(rdm.shape)

    # times = tmp_1['times']
    

# -------------------- Compute Linear Regression on Distances ----------------------------------------------------------
    train_size_list = [
                        0.02,   # very small train set
                        0.7,
                        0.75,
                        0.8, 
                        0.85,
                        0.9,
                        0.98,   # very big train set
                    ]
    print(train_size_list)              

    for size in train_size_list:
        all_scores = {}
        all_adj_scores = {}

        for metric in dist_metric:
            rdm = dist_metric[metric]

            r2_scores_w2v = []
            adj_r2_scores_w2v = []

            for i in range(rdm.shape[2]):    # loop across time_points
                brain_dist_vec = rdm[:,:, i]
                print(brain_dist_vec.shape)
        
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(w2v_rdm_vec,
                                                    brain_dist_vec, 
                                                    test_size=None,
                                                    train_size=size,
                                                    random_state=42, 
                                                    shuffle=True, # ?
                                                    stratify=None, # stratified fashion, based on class (input : array-like)
                                                )
                print(X_train.shape)
                print(y_train.shape)
                print(X_test.shape)
                print(y_test.shape)

                X_train = X_train[:,None]
                X_test = X_test[:,None]

                
                linear_reg = LinearRegression(fit_intercept=True,  # Whether to calculate the intercept for this model.
                                                copy_X=True,      # If True, X will be copied; else, it may be overwritten.
                                                n_jobs=-1,        # -1 means using all processors
                                                positive=False,   # When set to True, forces the coefficients to be positive
                                                )

                linear_reg.fit(X_train, y_train)
                y_predLR = linear_reg.predict(X_test)

                r2 = r2_score(y_test.squeeze(), y_predLR.squeeze())
                print(f"R^2 Score LR for [0:{i}]:", round(r2, 6))
                adj_r2 = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
                r2_scores_w2v.append(r2)
                adj_r2_scores_w2v.append(adj_r2)   

                # Store errors
                errors = y_test - y_predLR
                # sns.histplot(errors, legend=False)
                print('MAE:', metrics.mean_absolute_error(np.exp(y_test), np.exp(y_predLR)))   
                print('MAE:', metrics.mean_absolute_error(y_test, y_predLR))   # error absolute
                print('MSE:', metrics.mean_squared_error(y_test, y_predLR))    # error au carré
                print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_predLR)))  # racine carrée

            all_scores[metric] = r2_scores_w2v
            all_adj_scores[metric] = adj_r2_scores_w2v
            

            # R² scores plots
            plt.figure()
            plt.scatter(y_test, y_predLR)  # Plot the test values against the predicted ones
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title(f'predict vs true, {metric}, common stim set, train_size {size}')
            plt.grid(True)
            plt.savefig(f'{fig_path}/linear_reg_w2v_brain_{metric}_common_set_scatter_TrainSize_{size}_TEST.png')
            # plt.close()

            plt.figure()
            plt.plot(times, r2_scores_w2v)  # Plot the test values against the predicted ones
            plt.xlabel('Time (s)')
            plt.ylabel('R² score')
            plt.title(f'R² score, {metric}, common stim set, train_size {size}')
            plt.grid(True)
            plt.savefig(f'{fig_path}/linear_reg_w2v_brain_{metric}_common_set_curve_TrainSize_{size}_TEST.png')
            # plt.close()



            # Adjusted R² scores plots
            plt.figure()
            plt.plot(times, adj_r2_scores_w2v)  # Plot the test values against the predicted ones
            plt.xlabel('Time (s)')
            plt.ylabel('Adjusted R² score')
            plt.title(f'Adjusted R² score, {metric}, common stim set, train_size {size}')
            plt.grid(True)
            plt.savefig(f'{fig_path}/linear_reg_w2v_brain_adj_{metric}_common_set_curve_TrainSize_{size}_TEST.png')
            # plt.close()
