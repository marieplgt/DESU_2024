
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

from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import TweedieRegressor
from sklearn.model_selection import LearningCurveDisplay, learning_curve

def dircreate(dirs_to_create):
    """
    Create directories if they don't exist.

    :param dirs_to_create: Directory, or list of directories to create
    :type dirs_to_create: strings of path
    :return: Nothing, create the directories in the filesystem

    """
    if not op.exists(dirs_to_create):
        os.makedirs(dirs_to_create)


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
    


    # -------------------- Load W2V Distances --------------------------------------------------------

    # old rdm
    w2v_dir = '/media/sf_sharedVM/github_repo/Aud2Sem_MEG/code/meg_analysis_code/cf_pipelines/pipelines/derivatives/models/semantic/'
    common_w2v_cos_file = os.path.join(w2v_dir, 'common_w2v_concatenate_all_rdms_cosine.hdf5')

    # models_dir = '/media/marieplgt/T7/docs/PHD/Aud2sem_project/AUD2SEM_BIDS/derivatives/stimuli_models'
    # common_w2v_cos_file = os.path.join(models_dir, 'w2v/w2v_preproc-concat_method-cosine_rdms.hdf5')
    tmp = sst.rdm.load_rdms_h5py(common_w2v_cos_file)

    w2v_rdm = tmp['rdms_x']      # vector form : (n_pairs, n_features, n_rdms) =(x, 1, 1)
    print(w2v_rdm.shape)

    # extract common stim set
    w2v_rdm_vec = w2v_rdm[:,:,0]
    print(f"w2v : {w2v_rdm_vec.shape}")
   

    # -------------------- Load Brain Distances --------------------------------------------------------

    # Based on averaged epochs across 8 subjects
            
    brain_rdms_crosslation_dir = '/media/marieplgt/T7/docs/PHD/Aud2sem_project/AUD2SEM_BIDS/derivatives/brain_rdms'
    common_w2v_crosslation_file = os.path.join(brain_rdms_crosslation_dir, "sub-01_03_04_05_06_07_08_09_13_ch-meg_stim-comm_filt-LP_70_Hz_movavg-0_1_resamp-200_Hz_rdms-crosslation_cov-mne_empirical_cvscheme-cvloo.hdf5")
    
    brain_rdms_crossnobis_dir = '/media/marieplgt/T7/docs/PHD/Aud2sem_project/AUD2SEM_BIDS/derivatives/brain_rdms'
    common_w2v_crossnobis_file = os.path.join(brain_rdms_crossnobis_dir, "sub-01_03_04_05_06_07_08_09_13_ch-meg_stim-comm_filt-LP_70_Hz_movavg-0_1_resamp-200_Hz_rdms-crossnobis_cov-mne_empirical_cvscheme-cvloo.hdf5")

    # crosslation distances
    tmp_1 = sst.rdm.load_rdms_h5py(common_w2v_crosslation_file)
    # print(tmp_1.keys())
    crosslation_rdm = tmp_1['rdms_x']      # vector form : (n_pairs, n_features, n_rdms)
    print(crosslation_rdm.shape)

    # crossnobis distances
    tmp_2 = sst.rdm.load_rdms_h5py(common_w2v_crossnobis_file)
    # print(tmp_2.keys())
    crossnobis_rdm = tmp_2['rdms_x']      # vector form : (n_pairs, n_features, n_rdms)
    # print(crossnobis_rdm.shape)

    dist_metric = {'crosslation' : crosslation_rdm,
                    'crossnobis': crossnobis_rdm,
                    }

    # # print(dist_metric.keys())
    # for metric in dist_metric.keys():
    #     rdm = dist_metric[metric]
    #     print(rdm.shape)

    times = tmp_1['times']
    # size = 0.1

    # find best model
    pipe = Pipeline([
        ('estimator', LinearRegression())  # Placeholder estimator
    ])

    param_grid = [
        {
            'estimator': [LinearRegression()],  # Model 1: Linear Regression
        },
        {
            'estimator': [RidgeCV()],  # Model 2: Ridge CV
            'estimator__scoring' : ["r2", "explained_variance", "neg_mean_squared_error"],
        },
        {
            'estimator': [BayesianRidge()],  # Model 3: Bayesian Ridge
            'estimator__alpha_1': [1e-6, 1e-5, 1e-4],  
            'estimator__alpha_2': [1e-6, 1e-5, 1e-4],
            'estimator__lambda_1': [1e-6, 1e-5, 1e-4],
            'estimator__lambda_2': [1e-6, 1e-5, 1e-4],
            'estimator__lambda_init': [0.8, 0.9, 1, 1.1, 1.2],
        },
        {
            'estimator': [TweedieRegressor()],
            'estimator__power': [0, 1, 1.2, 2, 3],
            'estimator__alpha': [0.0, 0.8, 1, 1.2],
            'estimator__link': ['auto', 'identity', 'log'],
            'estimator__solver': ['lbfgs', 'newton-cholesky'],
        },
    ]


    cv_folds = [5, 10, 15]  #, 20 #, 25, 30]

    for n_fold in cv_folds:
        print(f"Testing with {cv_folds} folds...")
        print(f'current nb of cv : {n_fold}')

        kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)

        for metric in dist_metric:
            print(metric)
            rdm = dist_metric[metric]

            r2_scores_w2v = []
            adj_r2_scores_w2v = []
            
            for i in range(rdm.shape[2]):    # loop across time_points
                brain_dist_vec = rdm[:,:, i]
                print(brain_dist_vec.shape)

                # Store cross-validation scores to average
                fold_r2_scores = []
                fold_adj_r2_scores = []

                for train_index, test_index in kf.split(w2v_rdm_vec):
                    X_train, X_test = w2v_rdm_vec[train_index], w2v_rdm_vec[test_index]
                    y_train, y_test = brain_dist_vec[train_index], brain_dist_vec[test_index]

                    print(X_train.shape)
                    y_train = np.ravel(y_train)
                    print(y_train.shape)

                    print(X_test.shape)
                    y_test = np.ravel(y_test)
                    print(y_test.shape)


                    # Set up the GridSearchCV with cross-validation (minimum, seems more important to cv when splitting)
                    grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring=None, n_jobs=-1)

                    grid_search.fit(X_train, y_train)           # search the best model at each time point !!
                    y_predLR = grid_search.predict(X_test)

                    print("Best Model:", grid_search.best_estimator_)
                    print("Best Parameters:", grid_search.best_params_)
                    print("Best CV Score:", grid_search.best_score_)
                    tmp = grid_search.best_estimator_
                    model_name = str(tmp['estimator'])

                    r2 = grid_search.score(X_test, y_test)
                    print(f"Test Score: {r2:.4f}")
                    # adjusted R²
                    adj_r2 = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

                    fold_r2_scores.append(r2)
                    fold_adj_r2_scores.append(adj_r2)   

                    # # Store errors
                    # errors = y_test - y_predLR

                    # # learning curve at each time point..
                    # train_sizes, train_scores, test_scores = learning_curve(
                    #                                             grid_search, X_train, y_train)
                    # display = LearningCurveDisplay(train_sizes=train_sizes, train_scores=train_scores, 
                    #                                 test_scores=test_scores, score_name="Score")
                    # display.plot()
                    # # plt.show()

                r2_scores_w2v.append(np.mean(fold_r2_scores))
                adj_r2_scores_w2v.append(np.mean(fold_adj_r2_scores))


            # R² scores plots
            plt.figure()
            plt.scatter(y_test, y_predLR)  # Plot the test values against the predicted ones
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title(f'predict vs true, {metric}, common stim set, cv {n_fold}, {model_name}')
            plt.grid(True)
            plt.savefig(f'{fig_path}/linear_reg_w2v_brain_{metric}_common_set_scatter_TrainSize_cv_{n_fold}_bestmodel.png')
            # plt.close()

            plt.figure()
            plt.plot(times, r2_scores_w2v)  # Plot the test values against the predicted ones
            plt.xlabel('Time (s)')
            plt.ylabel('R² score')
            plt.title(f'R² score, {metric}, common stim set, cv {n_fold}, {model_name}')
            plt.grid(True)
            plt.savefig(f'{fig_path}/linear_reg_w2v_brain_{metric}_common_set_curve_TrainSize_cv_{n_fold}_bestmodel.png')
            # plt.close()
        
# %% model by model analysis

param_grid_linear = {}
param_grid_ridge = {
    'scoring': ["r2", "explained_variance", "neg_mean_squared_error"]
}
param_grid_bayesian = {
    'alpha_1': [1e-6, 1e-5, 1e-4],
    'alpha_2': [1e-6, 1e-5, 1e-4],
    'lambda_1': [1e-6, 1e-5, 1e-4],
    'lambda_2': [1e-6, 1e-5, 1e-4],
    'lambda_init': [0.8, 0.9, 1, 1.1, 1.2],
}
param_grid_tweedie = {
    'power': [0, 1, 1.2, 2, 3],
    'alpha': [0.0, 0.8, 1, 1.2],
    'link': ['auto', 'identity', 'log'],
    'solver': ['lbfgs', 'newton-cholesky'],
}

# Define your models
models = {
    'Linear Regression': (LinearRegression(), param_grid_linear),
    'Ridge CV': (RidgeCV(), param_grid_ridge),
    'Bayesian Ridge': (BayesianRidge(), param_grid_bayesian),
    'Tweedie Regressor': (TweedieRegressor(), param_grid_tweedie),
}


cv_folds = [5, 10, 15] #, 20, 25, 30]

for n_fold in cv_folds:
    print(f"Testing with {cv_folds} folds...")
    print(f'current nb of cv : {n_fold}')

    kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)

    for metric in dist_metric:
        print(metric)
        rdm = dist_metric[metric]

        # Perform cross-validation for each model
        for model_name, (model, param_grid) in models.items():
            
            r2_scores_w2v = []
            adj_r2_scores_w2v = []
        
            for i in range(rdm.shape[2]):    # loop across time_points
                brain_dist_vec = rdm[:,:, i]
                print(brain_dist_vec.shape)

                # Store cross-validation scores to average
                fold_r2_scores = []
                fold_adj_r2_scores = []

                for train_index, test_index in kf.split(w2v_rdm_vec):
                    X_train, X_test = w2v_rdm_vec[train_index], w2v_rdm_vec[test_index]
                    y_train, y_test = brain_dist_vec[train_index], brain_dist_vec[test_index]

                    print(X_train.shape)
                    y_train = np.ravel(y_train)
                    print(y_train.shape)

                    print(X_test.shape)
                    y_test = np.ravel(y_test)
                    print(y_test.shape)


                    print(f"\nTesting model: {model_name}")
                    
                    grid_search = GridSearchCV(model, param_grid, cv=kf, scoring=None, n_jobs=-1)

                    grid_search.fit(X_train, y_train)
                    y_predLR = grid_search.predict(X_test)

                    print("Best Model:", grid_search.best_estimator_)
                    print("Best Parameters:", grid_search.best_params_)
                    print("Best CV Score:", grid_search.best_score_)


                    r2 = grid_search.score(X_test, y_test)
                    print(f"Test Score: {r2:.4f}")
                    # adjusted R²
                    adj_r2 = 1 - (1-r2)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)

                    fold_r2_scores.append(r2)
                    fold_adj_r2_scores.append(adj_r2)   

                    # # Store errors
                    # errors = y_test - y_predLR

                    # # learning curve at each time point..
                    # train_sizes, train_scores, test_scores = learning_curve(
                    #                                             grid_search, X_train, y_train)
                    # display = LearningCurveDisplay(train_sizes=train_sizes, train_scores=train_scores, 
                    #                                 test_scores=test_scores, score_name="Score")
                    # display.plot()
                    # # plt.show()

                r2_scores_w2v.append(np.mean(fold_r2_scores))
                adj_r2_scores_w2v.append(np.mean(fold_adj_r2_scores))


            # R² scores plots
            plt.figure()
            plt.scatter(y_test, y_predLR)  # Plot the test values against the predicted ones
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title(f'predict vs true, {metric}, common stim set, cv {n_fold}, {model_name}')
            plt.grid(True)
            plt.savefig(f'{fig_path}/linear_reg_w2v_brain_{metric}_common_set_scatter_TrainSize_cv_{n_fold}_{model_name}.png')
            # plt.close()

            plt.figure()
            plt.plot(times, r2_scores_w2v)  # Plot the test values against the predicted ones
            plt.xlabel('Time (s)')
            plt.ylabel('R² score')
            plt.title(f'R² score, {metric}, common stim set, cv {n_fold}, {model_name}')
            plt.grid(True)
            plt.savefig(f'{fig_path}/linear_reg_w2v_brain_{metric}_common_set_curve_TrainSize_cv_{n_fold}_{model_name}.png')
            # plt.close()
            