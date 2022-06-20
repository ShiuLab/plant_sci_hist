'''
For creating Word2Vec embedding-based text classification model

6/18/22 [Shiu] When getting bi and trigrams, min_count was hard coded to 5,
        instead of using the config file values. Rerun.
6/15/22 Created by Shiu.
'''

## for data
import argparse
import json
import pandas as pd
import numpy as np
import pickle
import sys
import itertools
from pathlib import Path

from sklearn import model_selection, metrics

## for word embedding with w2v
import gensim

## for deep learning
from tensorflow.keras import models, layers, callbacks, preprocessing

def read_configs(config_file):
  """Read configuration file and return a config_dict"""
  # required
  config_dict = {'lang_model':0,
                 'proj_dir':0,
                 'work_dir':0,
                 'corpus_combo_file':0,
                 'rand_state':0,
                 'w2v_param':0,}

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


def split_train_validate_test(corpus_combo_file, rand_state):
  '''Load data and split train, validation, test subsets for the cleaned texts
  Args:
    corpus_combo_file (str): path to the json data file
    rand_state (int): for reproducibility
  Return:
    train, valid, test (pandas dataframes): training, validation, testing sets
  '''
  # Load json file
  with corpus_combo_file.open("r+") as f:
      corpus_combo_json = json.load(f)

  # Convert json back to dataframe
  corpus_combo = pd.read_json(corpus_combo_json)

  # Cleaned corpus
  corpus = corpus_combo[['label','txt_clean']]

  # Split train test
  train, test = model_selection.train_test_split(corpus, 
      test_size=0.2, stratify=corpus['label'], random_state=rand_state)

  # Split train validate
  train, valid = model_selection.train_test_split(train, 
      test_size=0.25, stratify=train['label'], random_state=rand_state)

  X_train = train['txt_clean']
  X_valid = valid['txt_clean']
  X_test  = test['txt_clean']
  y_train = train['label']
  y_valid = valid['label']
  y_test  = test['label']

  print(f"    size: train={X_train.shape}, valid={X_valid.shape}," +\
        f" test={X_test.shape}")

  return [X_train, X_valid, X_test, y_train, y_valid, y_test]
  
def get_hyperparameters(w2v_param):
  ''' Return a list with hyperparameters based on the passed dictionary
  Adopted from:
    https://stackoverflow.com/questions/38721847/how-to-generate-all-combination-from-values-in-dict-of-lists-in-python
  Args:
    param (dict): a dictionary specified in the config.txt file.
  Return:
    param_list (list): a nested list of hyperparameters 
  '''
  print(w2v_param)
  keys, values = zip(*w2v_param.items())
  param_list = [v for v in itertools.product(*values)]
  
  return keys, param_list

def get_unigram(corpus):
  unigram = []
  for txt in corpus:
    lst_words = txt.split()
    unigram.append(lst_words)

  return unigram

def get_ngram(X_train, X_valid, X_test, ngram, min_count):

  uni_train = get_unigram(X_train)
  uni_valid = get_unigram(X_valid)
  uni_test  = get_unigram(X_test)

  if ngram == 1:
    return uni_train, uni_valid, uni_test
  # ngram >1
  else:
    # Get bigrams
    bigrams_detector  = gensim.models.phrases.Phrases(
                    uni_train, delimiter=" ", min_count=min_count, threshold=10)
    bigrams_detector  = gensim.models.phrases.Phraser(bigrams_detector)
    bi_train = list(bigrams_detector[uni_train])
    bi_valid = list(bigrams_detector[uni_valid])
    bi_test  = list(bigrams_detector[uni_test])

    # Return bigrams
    if ngram == 2:
      return bi_train, bi_valid, bi_test

    # Get trigrams and return them
    elif ngram == 3:
      trigrams_detector = gensim.models.phrases.Phrases(
                      bigrams_detector[uni_train], delimiter=" ", 
                      min_count=min_count, threshold=10)
      trigrams_detector = gensim.models.phrases.Phraser(trigrams_detector)
      tri_train = list(trigrams_detector[bi_train])
      tri_valid = list(trigrams_detector[bi_valid])
      tri_test  = list(trigrams_detector[bi_test])
      return tri_train, tri_valid, tri_test
    
    else:
      print('ERR: ngram cannot be larger than 3. QUIT!')
      sys.exit(0)


def get_w2v_model(X_train, X_valid, X_test, param, rand_state):
  '''Get ngram lists and w2v model
  Args:
  Return:
  '''
  [min_count, window, ngram] = param

  ngram_train, ngram_valid, ngram_test = get_ngram(X_train, X_valid, X_test, 
                                                  ngram, min_count)

  # Check if w2v model is already generated
  model_w2v_name = work_dir / f"model_cln_w2v_{min_count}-{window}-{ngram}"

  if model_w2v_name.is_file():
    print("  load the w2v model")
    with open(work_dir / model_w2v_name, "rb") as f:
        model_w2v = pickle.load(f)
  else:
    print("  geneate and save w2v model")
    model_w2v = gensim.models.Word2Vec(ngram_train, vector_size=300, 
                                      window=window, min_count=min_count, 
                                      sg=1, epochs=30, seed=rand_state)
    
    with open(model_w2v_name, "wb") as f:
      pickle.dump(model_w2v, f)

  return model_w2v, model_w2v_name, ngram_train, ngram_valid, ngram_test


def train_tokenizer(corpus):
  '''Train a tokenizer
  Args:
    corpus (list): a nested list of word lists
  Return:
    tokenizer (keras.preprocessing.text.Tokenizer): trained tokenizer
    dic_vocab_token (dict): token as key, index as value
  '''

  # intialize tokenizer
  # See: https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization
  # This is replaced by tf.keras.layers.TextVectorization
  tokenizer = preprocessing.text.Tokenizer(lower=True, split=' ', 
                oov_token="NaN", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')

  # tokenize corpus 
  tokenizer.fit_on_texts(corpus)

  # get token dictionary, with token as key, index number as value
  dic_vocab_token = tokenizer.word_index

  return tokenizer, dic_vocab_token


def get_embeddings(corpus, model_w2v, tokenizer, dic_vocab_token):

  # Transforms each text in texts to a sequence of integers.
  lst_text2seq = tokenizer.texts_to_sequences(corpus)

  # pad or trucate sequence
  X_w2v = preprocessing.sequence.pad_sequences(
                    lst_text2seq,      # List of sequences, each a list of ints 
                    maxlen=500,        # maximum length of all sequences
                    padding="post",    # 'pre' or 'post' 
                    truncating="post") # remove values from sequences > maxlen

  ## start the matrix (length of vocabulary x vector size) with all 0s

  embeddings = np.zeros((len(dic_vocab_token)+1, 300))
  not_in_emb = {}
  for word, idx in dic_vocab_token.items():
      ## update the row with vector
      try:
          embeddings[idx] =  model_w2v.wv[word]
      ## if word not in model then skip and the row stays all 0s
      except KeyError:
          not_in_emb[word] = 1

  return embeddings, X_w2v


def get_w2v_emb_model(embeddings):
  '''Build a deep learning model with Word2Vec embeddings
  Args:
    embeddings
  '''

  ## code attention layer
  def attention_layer(inputs, neurons):
    x = layers.Permute((2,1))(inputs)
    x = layers.Dense(neurons, activation="softmax")(x)
    x = layers.Permute((2,1), name="attention")(x)
    x = layers.multiply([inputs, x])
    return x

  ## input
  x_in = layers.Input(shape=(500,)) ## embedding
  x = layers.Embedding(input_dim=embeddings.shape[0],  
                      output_dim=embeddings.shape[1], 
                      weights=[embeddings],
                      input_length=500, trainable=False)(x_in)

  ## apply attention
  x = attention_layer(x, neurons=500)

  ## 2 layers of bidirectional lstm
  x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2, 
                          return_sequences=True))(x)
  x = layers.Bidirectional(layers.LSTM(units=15, dropout=0.2))(x)

  ## final dense layers
  x = layers.Dense(64, activation='relu')(x)
  y_out = layers.Dense(2, activation='softmax')(x)

  ## Initialize and compile model
  model = models.Model(x_in, y_out)
  model.compile(loss='sparse_categorical_crossentropy',
                optimizer='adam', 
                metrics=['accuracy'])

  return model


def run_main_function():

  # Split train/validate/test for cleaned text
  #   Will not focus on original due to issues with non-alphanumeric characters
  #   and stop words.
  print("\nRead file and split train/validate/test...")
  subsets = split_train_validate_test(corpus_combo_file, rand_state)

  # get w2c parameter list
  #   [min_count, window, ngram]
  param_keys, param_list  = get_hyperparameters(w2v_param)

  # iterate through different parameters
  with open(work_dir / f"scores_cln_w2v", "w") as f:
    f.write("run\ttxt_flag\tlang_model\tparameters\tvalidate_f1\t" +\
            "test_f1\tmodel_dir\n")
    run_num = 0
    for param in param_list:
      print(f"\n## param: {param}")
      best_score, model_dir, test_score = run_pipeline(param, subsets)

      f.write(f"{run_num}\tcln\t{lang_model}\t{str(param)}\t"+\
              f"{best_score}\t{test_score}\t{model_dir}\n")

      run_num += 1


def run_pipeline(param, subsets):
  '''Carry out the major steps'''

  rand_state = config_dict['rand_state']

  [X_train, X_valid, X_test, y_train, y_valid, y_test] = subsets

  # Get list of ngrams and w2v model
  print("  get list of ngrams and w2v model")
  model_w2v, model_w2v_name, ngram_train, ngram_valid, ngram_test = \
                      get_w2v_model(X_train, X_valid, X_test, param, rand_state)
  
  # Train tokenizer
  print("  train tokenizer")
  tokenizer, dic_vocab_token = train_tokenizer(ngram_train)

  # Get embeddings
  print("  get embeddings")
  embeddings, X_train_w2v = get_embeddings(ngram_train, model_w2v, 
                                                    tokenizer, dic_vocab_token)
  _, X_valid_w2v = get_embeddings(ngram_valid, model_w2v, 
                                                    tokenizer, dic_vocab_token)
  _ , X_test_w2v  = get_embeddings(ngram_test , model_w2v, 
                                                    tokenizer, dic_vocab_token)

  # Model checkpoint path and output model file name
  cp_filepath  = Path(str(model_w2v_name) + "_dnn")

  # Load model if exists
  if cp_filepath.is_dir():
    print("  load model in:", cp_filepath)
    model_emb = models.load_model(cp_filepath)

  # Train and save model if not
  else:
    print("  train model")
    model_emb    = get_w2v_emb_model(embeddings)

    # setup check points
    callback_es  = callbacks.EarlyStopping(monitor='val_loss', patience=5)
    callback_mcp = callbacks.ModelCheckpoint(filepath=cp_filepath, mode='max', 
            save_weights_only=False, monitor='val_accuracy', save_best_only=True)

    # Train model
    history = model_emb.fit(x=X_train_w2v, y=y_train, batch_size=256, 
                            epochs=20, shuffle=True, verbose=1, 
                            validation_data=(X_valid_w2v, y_valid), 
                            callbacks=[callback_es, callback_mcp])

  print("  get validation f1 score")
  y_valid_pred_prob = model_emb.predict(X_valid_w2v)
  dic_y_mapping = {n:label for n,label in enumerate(np.unique(y_valid))}
  y_valid_pred = [dic_y_mapping[np.argmax(pred)] for pred in y_valid_pred_prob]
  best_score = metrics.f1_score(y_valid, y_valid_pred)
  print("    ", best_score)

  print("  get testing f1 score")
  y_test_pred_prob = model_emb.predict(X_test_w2v)
  dic_y_mapping = {n:label for n,label in enumerate(np.unique(y_test))}
  y_test_pred = [dic_y_mapping[np.argmax(pred)] for pred in y_test_pred_prob]
  test_score = metrics.f1_score(y_test, y_test_pred)
  print("    ", test_score)

  # provide some space between runs
  print('\n')

  return best_score, cp_filepath, test_score

################################################################################


argparser = argparse.ArgumentParser()
argparser.add_argument('-c', '--config',
                      help='Configuration file', required=True)
args = argparser.parse_args()

config_file = Path(args.config)

print("\nRead configuration file...")
config_dict = read_configs(config_file)

# Declare config parameters as global variables
proj_dir          = Path(config_dict['proj_dir'])
work_dir          = proj_dir / config_dict['work_dir']
corpus_combo_file = work_dir / config_dict['corpus_combo_file']
lang_model        = config_dict['lang_model']
rand_state        = config_dict['rand_state']
w2v_param         = config_dict['w2v_param']

run_main_function()


