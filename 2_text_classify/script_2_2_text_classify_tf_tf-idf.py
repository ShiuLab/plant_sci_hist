'''
For building text classification model based on tf and tf-idf using either
original corpus or cleaned one.

07/02/22 [Shiu]: When trying to interpret the saved model, run into issue
  loading the joblib .save files. 
06/13/22 [Shiu]: Create config.txt and set up all config info in the file.
  Also, realized that some global variables need not to be passed around (e.g.,
  config_dict). But did not fix that.
06/13/22 [Shiu]: Include ways to run tf-idf through command line argument.
06/13/22 [Shiu]: Change to v.2. Through some prelim runs, fond that stop words 
  and p_thresholds have essentially no effect (impact F1 by <0.0015) so they 
  are not considered. The results are in the _run_stop_p folder in work_dir.
'''

## for data
import argparse
import json
import pickle
import pandas as pd
import numpy as np
import joblib
import sys
from os import chdir
from pathlib import Path

## for bag-of-words
from sklearn import feature_extraction, feature_selection, metrics
from sklearn import model_selection
from xgboost import XGBClassifier

def read_configs(config_file):
  """Read configuration file and return a config_dict"""
  # required
  config_dict = {'lang_model':0,
                 'proj_dir':0,
                 'work_dir':0,
                 'corpus_combo_file':0,
                 'rand_state':0,
                 'vec_param':0,
                 'p_threshold':0,
                 'xg_param':0,
                 'n_splits':0,
                 'xg_param_comb':0,
                 'n_jobs':0,}

  # Read config file and fill in the dictionary
  with open(config_file, 'r') as f:
    configs     = f.readlines()
    for config in configs:
      if config.strip() == "" or config[0] == "#":
        pass
      else:
        config = config.strip().split("=")
        if config[0] in config_dict:
          config_dict[config[0]] = eval(config[1])

  # Check if any config missing
  missing = 0
  for config in config_dict:
    if config_dict[config] == 0:
      print("  missing:", config)
      missing += 1
    else:
      print("  ", config, "=", config_dict[config])

  if missing == 0:
    print("  all config available")
  else:
    print("  missing config, QUIT!")
    sys.exit(0)

  return config_dict


def split_train_test(corpus_combo_file, rand_state):
  '''Load data and split train test
  Args:
    corpus_combo_file (str): path to the json data file
    rand_state (int): for reproducibility
  Return:
    train_ori, test_ori, train_cln, test_cln (pandas dataframes): for the
      original and clean texts, training and testing splits.
  '''
  # Load json file
  with corpus_combo_file.open("r+") as f:
      corpus_combo_json = json.load(f)

  # Convert json back to dataframe
  corpus_combo = pd.read_json(corpus_combo_json)

  corpus_ori = corpus_combo[['label','txt']]
  train_ori, test_ori = model_selection.train_test_split(corpus_ori, 
      test_size=0.2, stratify=corpus_ori['label'], random_state=rand_state)

  # Cleaned corpus
  corpus_cln = corpus_combo[['label','txt_clean']]
  corpus_cln.rename(columns={'txt_clean': 'txt'}) # make col names consistent
  train_cln, test_cln = model_selection.train_test_split(corpus_cln, 
      test_size=0.2, stratify=corpus_cln['label'], random_state=rand_state)

  return train_ori, test_ori, train_cln, test_cln

def get_hyperparameters(vec_param, p_threshold):
  ''' Return a list with hyperparameters based on the passed dictionary
  Args:
    vec_param: the dictionary specified in the config.txt file for vectorizer.
  Return:
    param_list (list): a nested list of hyperparameters in the order of
      max_feature, ngram_range, and p_threshold
  '''
  param_list = []
  for i in vec_param['max_features']:
    i = int(i) # max_features has to be interger.
    for j in vec_param['ngram_range']:
      param_list.append([i, j, p_threshold])
  
  return param_list

def extract_feat(X_train, param=[], lang_model="",  vocab=""):
  '''Extracting features as term frequencies
  Args:
    X_train (pandas series): the txt column in the training data frame
    param (list): contains max_features, ngram_range, p_threshold
    lang_model (str): tf or tf-idf
    vocab (list): a list of features to fit.
  Returns:
    vectorizer (sklearn.feature_extraction.text.CountVectorizer) 
    X_train (pandas series): the transformed X_train
  '''
  # vectorizerd term frequencies
  if vocab == "":
    [max_features, ngram_range, _] = param
    if lang_model == 'tf':
      vectorizer = feature_extraction.text.CountVectorizer(
                            max_features = max_features, 
                            ngram_range  = ngram_range)
    elif lang_model == "tf-idf":
      vectorizer = feature_extraction.text.TfidfVectorizer(
                            max_features = max_features, 
                            ngram_range  = ngram_range)
    else:
      print("ERROR: Unknown lang_model specified. QUIT!")
      sys.exit(0)
  else:
    vectorizer = feature_extraction.text.CountVectorizer(vocabulary=vocab)

  # fit the vectorizer with training corpus
  vectorizer.fit(X_train)

  # transform the training corpus
  X_train_vec = vectorizer.transform(X_train)

  return vectorizer, X_train_vec

def select_feat(X_train, y_train, vectorizer, p_threshold):
  '''Select features based on chi-square test results
  Args:
    X_train (pandas series): the txt column in the training data frame
    y_train (pandas series): the label column in the training data frame
    vecorizer: fitted with original X_train and returned from get_vectorizer()
    p_threshold (float): p is derived from chi-square test. Features with p <= 
      p_threshold_s are selected.
  Return:
    X_names (list): names of selected features
  '''
  y            = y_train
  X_names      = vectorizer.get_feature_names_out()
  dtf_features = pd.DataFrame()
  for cat in np.unique(y):
    _, p = feature_selection.chi2(X_train, y==cat)
    dtf_features = pd.concat([dtf_features, 
                pd.DataFrame({"feature":X_names, "p":p, "y":cat})])
    dtf_features = dtf_features.sort_values(
                ["y","p"], ascending=[True,False])
    dtf_features = dtf_features[dtf_features["p"] <= p_threshold]
  
  X_names = dtf_features["feature"].unique().tolist()

  return X_names

def run_xgboost(X_train, y_train, config_dict):
  '''Do hyperparameter tuning and cross-validation of XgBoost models
  Args:
    X_train (pandas dataframe): features
    y_train (pandas series): labels
    config_dict (dict): from read_config()
  Return:
    rand_search (RandomizedSearchCV): fitted obj
  '''

  rand_state = config_dict["rand_state"]
  param_grid = config_dict["xg_param"] 
  n_splits   = config_dict["n_splits"]
  param_comb = config_dict["xg_param_comb"]
  n_jobs     = config_dict["n_jobs"]

  # Initialize classifier
  # 06/11/2022: the silent parameter is deprecated, use verbosity=0
  xgb = XGBClassifier(learning_rate=0.02, 
                      n_estimators=600, 
                      objective='binary:logistic',
                      verbosity=1, 
                      nthread=1)
  # Initilize stratified k fold obj
  skf = model_selection.StratifiedKFold(n_splits=n_splits, 
                      shuffle = True, random_state = rand_state)
  # initiate randomized search CV obj
  rand_search = model_selection.RandomizedSearchCV(
                      xgb                , param_distributions = param_grid, 
                      n_iter = param_comb, scoring      = 'f1', 
                      n_jobs = n_jobs    , cv = skf.split(X_train,y_train), 
                      verbose = 3        , random_state =rand_state)
  # Train
  rand_search.fit(X_train, y_train)

  return rand_search


def run_main_function(work_dir, train, test, txt_flag, config_dict):

  #write training data to files
  print("  write training data to file for interpretation")
  train_file = work_dir / f"corpus_train_{txt_flag}.tsv.gz"
  train.to_csv(train_file, sep="\t", compression='gzip')

  # Get the training/testing corpus and labels
  if txt_flag == "ori":
    X_train = train['txt']
    X_test  = test['txt']
  else:
    X_train = train['txt_clean']
    X_test  = test['txt_clean']

  y_train = train['label']
  y_test  = test['label']

  # get vectorizer parameter list
  p_threshold = config_dict['p_threshold']
  param_list  = get_hyperparameters(config_dict['vec_param'], p_threshold)
  lang_model  = config_dict['lang_model']

  # iterate through different parameters
  with open(work_dir / f"scores_{txt_flag}", "w") as f:
    f.write("run\ttxt_flag\tlang_model\tparameters\tnum_feat\tcv_f1\t" +\
            "test_f1\tmodel_name\n")
    run_num = 0
    for param in param_list:
      print(f"\n## param: {param}")
      best_score, num_select, model_name, test_score = run_pipeline(
        work_dir, X_train, y_train, X_test, y_test, param, txt_flag, config_dict)

      f.write(f"{run_num}\t{txt_flag}\t{lang_model}\t{str(param)}\t"+\
              f"{num_select}\t{best_score}\t{test_score}\t{model_name}\n")

      run_num += 1


def run_pipeline(work_dir, X_train, y_train, X_test, y_test, param, txt_flag,
                 config_dict):
  '''Carry out the major steps'''

  # For saving files
  param_str  = \
      f"{int(param[0])}-{'to'.join(map(str,param[1]))}-{param[2]}"

  # Get vectorizer and fitted X_train
  print("  extract features by fitting a vectorizer")
  lang_model = config_dict['lang_model']
  vectorizer, X_train_vec = extract_feat(X_train, param, lang_model)
  print("    train dim:", X_train_vec.shape)

  # Get selected feature names
  print("  select features")
  p_threshold = config_dict['p_threshold']
  X_names     = select_feat(X_train_vec, y_train, vectorizer, p_threshold)
  num_select  = len(X_names)
  print('    total selected:', num_select)

  # Save the selected features
  X_names_file = work_dir / \
              f"corpus_{txt_flag}_{lang_model}_{param_str}_sel_featnames.json"
  with open(X_names_file, 'w+') as f:
      json.dump(pd.Series(X_names).to_json(), f)

  # Refit vectorizer with selected features and re-transform X_train
  print("  refit vectorizer with training data and transform")
  vectorizer_sel, X_train_vec_sel = extract_feat(X_train, vocab=X_names)
  print("    train dim:", X_train_vec_sel.shape)

  # Also apply the refitted vecorizer to testing data
  print("  transform testing data")
  X_test_vec_sel = vectorizer_sel.transform(X_test)
  print("    test dim:", X_test_vec_sel.shape)

  # Save the vectorized train/test data
  print("  save vectorizer and vectorized features")
  vec_sel_file = work_dir / f"corpus_{txt_flag}_{lang_model}_{param_str}_vec_sel.pkl"
  with open(vec_sel_file, 'wb') as f:
    pickle.dump(vectorizer_sel, f)

  # Convert scipy sparse arrays to dataframe
  X_train_vec_sel_df = pd.DataFrame.sparse.from_spmatrix(X_train_vec_sel, 
                                                         columns=X_names)
  # 10/7/25: discovered a bug here, I passed training data instead of
  # testing. So, not really testing the results. Ok, this is ONLY for export
  # purpose and was not used for evaluation. In the evaluation step below,
  # the correct testing vector sparse matrix is used. Man...
  #X_test_vec_sel_df  = pd.DataFrame.sparse.from_spmatrix(X_train_vec_sel, 
  #                                                       columns=X_names)
  X_test_vec_sel_df  = pd.DataFrame.sparse.from_spmatrix(X_test_vec_sel, 
                                                         columns=X_names) 

  # Specify file names 
  X_train_vec_sel_file = work_dir / \
              f"corpus_{txt_flag}_{lang_model}_{param_str}_train_vec_sel.json"
  X_test_vec_sel_file  = work_dir / \
              f"corpus_{txt_flag}_{lang_model}_{param_str}_test_vec_sel.json"
  
  # Write json files
  with open(X_train_vec_sel_file, 'w+') as f:
      json.dump(X_train_vec_sel_df.to_json(), f)
  with open(X_test_vec_sel_file, 'w+') as f:
      json.dump(X_test_vec_sel_df.to_json(), f)

  # Check if model already exist
  model_name = work_dir / f'model_{txt_flag}_{lang_model}_{param_str}.sav'
  #if model_name.is_file():
  #  print("  load existing model")
  #  rand_search = joblib.load(model_name)
  #else:

  # Get xgboost model and cv results
  print("  cross-validation and tuning with xgboost")
  rand_search = run_xgboost(X_train_vec_sel, y_train, config_dict)
  # Save the best model
  print("  save model")
  best_est = rand_search.best_estimator_
  joblib.dump(best_est, model_name)

  best_param = rand_search.best_params_
  best_score = rand_search.best_score_
  print("    best F1:", best_score)
  print("    best param:", best_param)

  # Get testing results: This is not for tuning/selection purpose but because
  # X_test is transformed by vectorizer for each parameter combination. If
  # the testing set is not evaluated now, things just get too complicated.
  # Keep in mind that the testing F1s will not be compared against each other.
  print("  get testing f1")
  y_pred = best_est.predict(X_test_vec_sel)
  test_score = metrics.f1_score(y_test, y_pred)

  # provide some space between runs
  print('\n')

  return best_score, num_select, model_name, test_score


################################################################################

if __name__ == "__main__":
  argparser = argparse.ArgumentParser()
  argparser.add_argument('-c', '--config',
                        help='Configuration file', required=True)
  args = argparser.parse_args()

  print("\nRead configuration file...")
  config_dict = read_configs(args.config)

  # Set up working directory and corpus file location
  proj_dir          = Path.home() / config_dict['proj_dir']
  work_dir          = proj_dir / config_dict['work_dir']
  corpus_combo_file = work_dir / config_dict['corpus_combo_file']

  # For reproducibility
  rand_state = config_dict['rand_state']

  # Split train/test for original and cleaned text
  print("\nRead file and split train/test...")
  train_ori, test_ori, train_cln, test_cln = split_train_test(
                                                corpus_combo_file, rand_state)

  print("\nRun main function with original data...")
  run_main_function(work_dir, train_ori, test_ori, "ori", config_dict)

  # Do not need to run on cleaned data
  #print("\nRun main function with cleaned data...")
  #run_main_function(work_dir, train_cln, test_cln, "cln", config_dict)


