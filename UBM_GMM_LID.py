import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
from tqdm.notebook import tqdm_notebook
from tabulate import tabulate

source_path = '<SOURCE_PATH>'
os.chdir(source_path)

## 1. Use some of the function as it is from GMM_LID.py
def GMM_load_trained_model(model_type, n_comps, cov_type, class_label):
  '''
    model_type: str -> {GMM or UBM-GMM}
    n_comps: int -> no. of components
    cov_type: str -> {diag or full}
    class_label: str -> languages ...
  '''
  filepath = f'results/{model_type}_{cov_type}_{n_comps}_{class_label}.pickle'
  model = pickle.load(open(source_path + filepath, 'rb'))
  return model

## For training a Universal Background Model
def GMM_training_single(train_df, n_comps, cov_type, class_label, random_seed = 42, model_type = "GMM"):
  filepath = f'results/{model_type}_{cov_type}_{n_comps}_{class_label}.pickle'

  if os.path.exists(source_path + filepath):
    return GMM_load_trained_model(model_type, n_comps, cov_type, class_label)

  # initialising gmm model
  model = GaussianMixture(n_components=n_comps, covariance_type=cov_type, random_state=random_seed, init_params='k-means++')
  # fitting the model with training data
  model.fit(train_df.values)

  # dumping/saving trained model
  pickle.dump(model, open(source_path + filepath, 'wb'))
  return model  # --> successful


# Map estimate or E-step of EM algorithm

def MAP_adapt(universal_gmm, train_df, cov_type, n_comps, class_label, max_iter=100, r=0.7, model_type = "UBM-GMM"):
  '''
    universal_gmm:-> gmm built on top of all the data combined
    train_df:-> DataFrame of a single language
    ---- will help in model saving
    cov_type:-> {diag or full} 
    n_comps:-> #clusters
    class_label:-> language of train_df
  '''
  filepath = f'results/{model_type}_{cov_type}_{n_comps}_{class_label}.pickle'
  if os.path.exists(source_path + filepath):
    return GMM_load_trained_model(model_type, n_comps, cov_type, class_label)

  # copy the initial model
  gmm = copy.deepcopy(universal_gmm)
  X = train_df # refrence not copy
  
  ## MAP Adaptation for means    
  for i in range(max_iter):
      n = np.sum(gmm.predict_proba(X), axis=0).reshape(-1, 1) # shape = (K, 1)
      X_tilde = (1 / n) * gmm.predict_proba(X).T.dot(X) # shape = (K, d) --> K = num_component
      alpha = (n / (n + r)).reshape(-1, 1) # shape = (K, 1)
      gmm.means_ = alpha * X_tilde + (1 - alpha) * gmm.means_

  # dumping/saving trained model
  pickle.dump(gmm, open(source_path + filepath, 'wb'))
  return gmm  # --> successful

# A Higher level abstraction functiion
def UBM_GMM_training_all(combined_df, train_dfs, n_comps, cov_type):
  ubm_gmm = GMM_training_single(combined_df, n_comps, cov_type, 'overall',
                                model_type = "UBM-GMM")
  class_specific_gmm = []
  for i in tqdm_notebook(range(len(train_dfs)), desc='specific gmm progress'):
      lang_i_gmm = MAP_adapt(ubm_gmm, train_dfs[i].values, cov_type, n_comps, languages[i])
      class_specific_gmm.append(lang_i_gmm)
  return class_specific_gmm      

## 3. Prediction dependencies
# prior (training) log probabilities for each classes
priors = [len(train_df_list[i])/total_samples for i in range(len(train_df_list))]
log_priors = [np.log(prob) for prob in priors]

def GMM_prediction(gmm_lists, test_df_list, y_test):
    '''This Function takes 
      gmm_lists :-> list of models of each class,
      test_df_list :-> list of test DataFrames,
      y_test :-> combined true test labels as input'''
    
    ## Before calculating log probabilities we need to combine all the data_frames on top of each other in one csv file
    #     concatenating df1 and df2 along rows
    #     vertical_concat = pd.concat([df1, df2], axis=0)
    test = pd.concat(test_df_list, axis=0)
    
    ## weightage log likelihood using gmm.score_samples(x)
    log_lists = [gmm.score_samples(test.values) for gmm in gmm_lists]
    
    ## Now predicting output based on build model
    y_pred = [] #predicted labels based on our model
    for i in range(len(test)):
        # we need to find argmax such that log probability is maximum
        index_of_log_max = 0
        for j in range(len(log_lists)):
            if(log_lists[j][i] + log_priors[j] > log_lists[index_of_log_max][i] + log_priors[index_of_log_max]):
                index_of_log_max = j
        y_pred.append(index_of_log_max)
    
    return y_test, y_pred

# There are some warnings as the datasets are not loaded in this python script