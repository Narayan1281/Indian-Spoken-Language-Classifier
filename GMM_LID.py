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
# os.listdir()

pb_train_path = source_path + 'pb_train/'
pb_test_path = source_path + 'pb_test/'
yt_test_path = source_path + 'yt_test/'

# Get all the training/test csv file names
language_file_names = [f for f in os.listdir(pb_train_path) if f.endswith('.csv')]
# print(language_file_names)

languages = [f[:3] for f in language_file_names] ## Names of all the language classes
print("all available languages are \n",languages)

## 1. Loading Datasets

# training dataset
os.chdir(pb_train_path)
train_df_list = [pd.read_csv(f,header=None,encoding='UTF-16') for f in language_file_names] # list of dataframes of each language
os.chdir(source_path)

# test dataset
os.chdir(pb_test_path)
pb_test_df_list = [pd.read_csv(f,header=None,encoding='UTF-16') for f in language_file_names] # list of dataframes of each language
os.chdir(source_path)

# Yt test dataset
os.chdir(yt_test_path)
yt_test_df_list = [pd.read_csv(f,header=None,encoding='UTF-16') for f in language_file_names] # list of dataframes of each language
os.chdir(source_path)


# combined true labels for these data
y_train = [] # true labels
for i in range(len(train_df_list)):
    for _ in range(len(train_df_list[i])):
        y_train.append(i)

# pb_test
y_pb_test = []
for i in range(len(pb_test_df_list)):
    for _ in range(len(pb_test_df_list[i])):
        y_pb_test.append(i)

# yt_test
y_yt_test = []
for i in range(len(yt_test_df_list)):
    for _ in range(len(yt_test_df_list[i])):
        y_yt_test.append(i)


## 2. Details about training datasets
lang_num_samples = []
total_samples = 0
head = ['languages', 'num_samples']
for i, df in enumerate(train_df_list):
  total_samples += len(df)
  lang_num_samples.append([languages[i], len(df)])

print(tabulate(lang_num_samples, headers=head, tablefmt="grid"))
print('#training samples: ', total_samples)

## 3. Preprocessing
## normalise the data samples in range [0, 1]
# before feeding to model
def normalise_df(df, minimums, maximums):
  normalised_ = (df - minimums)/(maximums - minimums)
  return normalised_

train_min_max_list = [[df.min(), df.max()] for df in train_df_list] # storing min and max value for training data samples, for all the features


# Prasar Bharti Training Dataset
normalised_train_df_list = [normalise_df(train_df_list[i], train_min_max_list[i][0], train_min_max_list[i][1]) \
                            for i in range(len(train_df_list))]

# Prasar Bharti Test Dataset
normalised_pb_test_df_list = [normalise_df(pb_test_df_list[i], train_min_max_list[i][0], train_min_max_list[i][1]) \
                              for i in range(len(pb_test_df_list))]

# YouTube Test Dataset
normalised_yt_test_df_list = [normalise_df(yt_test_df_list[i], train_min_max_list[i][0], train_min_max_list[i][1]) \
                              for i in range(len(yt_test_df_list))]

## 4. Model training/ loading trained model
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

def GMM_training_all(train_dfs, n_comps, cov_type, random_seed = 42):
  gmm_lists = []
  for i in tqdm_notebook(range(len(train_dfs)), desc="model training loop"):
    curr_model = GMM_training_single(train_dfs[i], n_comps, cov_type, languages[i])
    gmm_lists.append(curr_model)
  # print(f'********** training complete for ${cov_type} covariance with ${n_comps} components ******')
  return gmm_lists  


## 5. Prior probabilities of each classes
# prior (training) log probabilities for each classes
priors = [len(train_df_list[i])/total_samples for i in range(len(train_df_list))]
log_priors = [np.log(prob) for prob in priors]

## 6. prediction function of a gaussian mixture model

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
    
## the above functions can be used to train and test a GMM model