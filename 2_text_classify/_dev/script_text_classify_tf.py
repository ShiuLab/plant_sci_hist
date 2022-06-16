'''
For building text classification model based on term frequency
'''

## For reproducibility
rand_state = 20220609

## for data
import json
import pandas as pd
import numpy as np
import joblib
from os import chdir
from pathlib import Path

## for bag-of-words
from sklearn import feature_extraction, feature_selection, metrics
from sklearn import model_selection
from xgboost import XGBClassifier

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

def get_hyperparameters(stopw=0):
    ''' Return a dictionary with hyperparameters
    Args
      stopw (int): whether to rid of english stopwords (1) or not (0)
    Return:
      param_list (list): a nested list of hyperparameters in the order of
        max_feature, ngram_range, stop_words, and p_threshold
    '''
   
    param_grid = {"max_features": [1e4, 5e4, 1e5],
                  "ngram_range": [(1,1), (1,2), (1,3)],
                  "stop_words": [None],
                  "p_threshold": [1e-2, 1e-3, 1e-4, 1e-5]}
    #print(param_grid)
    if stopw:
        param_grid["stop_words"].append("english")

    param_list = []
    for i in param_grid['max_features']:
        for j in param_grid['ngram_range']:
            for k in param_grid['stop_words']:
                for l in param_grid['p_threshold']:
                    param_list.append([i, j, k, l])
    
    return param_list

def extract_feat(X_train, param=[], vocab=""):
    '''Extracting features as term frequencies
    Args:
      X_train (pandas series): the txt column in the training data frame
      param (list): contains max_features, ngram_range, stop_words, p_threshold
      vocab (list): a list of features to fit.
    Returns:
      vectorizer (sklearn.feature_extraction.text.CountVectorizer) 
      X_train (pandas series): the transformed X_train
    '''
    # vectorizerd term frequencies
    if vocab == "":
      [max_features, ngram_range, stop_words, _] = param
      max_features = int(max_features)
      vectorizer = feature_extraction.text.CountVectorizer(
                              max_features = max_features, 
                              ngram_range  = ngram_range,
                              stop_words   = stop_words)
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

def run_xgboost(X_train, y_train, rand_state):
    '''Do hyperparameter tuning and cross-validation of XgBoost models
    Args:
      X_train (pandas dataframe): features
      y_train (pandas series): labels
      rand_state (int): rand
    Return:
      rand_search (RandomizedSearchCV): fitted obj
    '''

    param_grid = {'min_child_weight': [1, 5, 10],
                  'gamma': [0.5, 1, 1.5, 2, 5],
                  'subsample': [0.6, 0.8, 1.0],
                  'colsample_bytree': [0.6, 0.8, 1.0],
                  'max_depth': [3, 4, 5]}
    folds       = 5
    param_comb  = 5
    n_jobs      = 14

    # Initialize classifier
    # 06/11/2022: the silent parameter is deprecated, use verbosity=0
    xgb = XGBClassifier(learning_rate=0.02, 
                        n_estimators=600, 
                        objective='binary:logistic',
                        verbosity=1, 
                        nthread=1)
    # Initilize stratified k fold obj
    skf = model_selection.StratifiedKFold(n_splits=folds, 
                        shuffle = True, random_state = rand_state)
    # initiate randomized search CV obj
    rand_search = model_selection.RandomizedSearchCV(
                        xgb                , param_distributions = param_grid, 
                        n_iter = param_comb, scoring      = 'f1', 
                        n_jobs = n_jobs    , cv = skf.split(X_train,y_train), 
                        verbose = 1        , random_state =rand_state)
    # Train
    rand_search.fit(X_train, y_train)

    return rand_search


def run_main_function(work_dir, train, test, txt_flag):

    # Get the training/testing corpus and labels
    X_train = train['txt']
    y_train = train['label']
    X_test  = test['txt']
    y_test  = test['label']

    # Whether stop_words parameter should be included or not
    stopw   = 0
    if txt_flag == "ori":
        stopw = 1
    param_list  = get_hyperparameters(stopw=stopw)
    
    # iterate through different parameters
    with open(work_dir / f"scores_{txt_flag}", "w") as f:
        f.write("run\ttxt_flag\tparameters\tmum_feat\tcv_f1\ttest_f1\tmodel_name\n")
        run_num = 0
        for param in param_list:
            print(f"\n#####\nparam: {param}")
            best_score, num_select, model_name, test_score = run_pipeline(
                work_dir, X_train, y_train, X_test, y_test, param, txt_flag)

            f.write(f"{run_num}\t{txt_flag}\t{str(param)}\t{num_select}\t"+\
                    f"{best_score}\t{test_score}\t{model_name}\n")

            run_num += 1


def run_pipeline(work_dir, X_train, y_train, X_test, y_test, param, txt_flag):
    '''Carry out the major steps'''

    # Get vectorizer and fitted X_train
    print("  extract features by fitting a vectorizer")
    vectorizer, X_train_vec = extract_feat(X_train, param=param)
    print("    train dim:", X_train_vec.shape)

    # Get selected feature names
    print("  select features")
    p_threshold = param[-1]
    X_names     = select_feat(X_train_vec, y_train, vectorizer, p_threshold)
    num_select  = len(X_names)
    print('    total selected:', num_select)

    # Refit vectorizer with selected features and re-transform X_train
    print("  refit vectorizer with training data and transform")
    vectorizer_sel, X_train_vec_sel = extract_feat(X_train, vocab=X_names)
    print("    train dim:", X_train_vec_sel.shape)

    # Also apply the refitted vecorizer to testing data
    print("  transform testing data")
    X_test_vec_sel = vectorizer_sel.transform(X_test)
    print("    test dim:", X_test_vec_sel.shape)

    # Get xgboost model and cv results
    print("  cross-validation and tuning with xgboost")
    rand_search = run_xgboost(X_train_vec_sel, y_train, rand_state)

    best_est   = rand_search.best_estimator_
    best_param = rand_search.best_params_
    best_score = rand_search.best_score_
    print("    best F1:", best_score)
    print("    best param:", best_param)

    # Save the best model
    param_str  = \
        f"{int(param[0])}-{'|'.join(map(str,param[1]))}-{param[2]}-{param[3]}"

    model_name = work_dir / f'model_{txt_flag}_{param_str}.sav'
    joblib.dump(best_est, model_name)

    # Get testing results: This is not for tuning/selection purpose but because
    # X_test is transformed by vectorizer for each parameter combination. If
    # the testing set is not evaluated now, things just get too complicated.
    # Keep in mind that the testing F1s will not be compared against each other.
    print("  Get testing f1")
    y_pred = best_est.predict(X_test_vec_sel)
    test_score = metrics.f1_score(y_test, y_pred)

    # provide some space between runs
    print('\n')

    return best_score, num_select, model_name, test_score


################################################################################

# Set up working directory and corpus file location
proj_dir          = Path('/home/shius/projects/plant_sci_hist')
work_dir          = proj_dir / "2_text_classify"
corpus_combo_file = work_dir / "corpus_combo"

# Split train/test for original and cleaned text
print("\nRead file and split train/test...")
train_ori, test_ori, train_cln, test_cln = split_train_test(
                                                corpus_combo_file, rand_state)

print("\nRun main function with original data...")
run_main_function(work_dir, train_ori, test_ori, "ori")

print("\nRun main function with cleaned data...")
run_main_function(work_dir, train_cln, test_cln, "cln")


