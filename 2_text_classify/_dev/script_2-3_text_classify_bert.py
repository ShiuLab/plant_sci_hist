'''
For creating BERT-based text classification model

6/17/22 Created by Shiu.
'''

import argparse
import json
import pandas as pd
import numpy as np
import pickle
import sys
import itertools
from pathlib import Path

from sklearn import model_selection, metrics

import transformers
from datasets import Dataset
from tokenizers import BertWordPieceTokenizer

import tensorflow as tf


def read_configs(config_file):
  """Read configuration file and return a config_dict"""
  # required
  config_dict = {'lang_model':0,
                 'proj_dir':0,
                 'work_dir':0,
                 'corpus_combo_file':0,
                 'rand_state':0,
                 'vocab_size':0,
                 'max_length':0,
                 'min_frequency':0,
                 'bert_param':0,}

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

  corpus = corpus_combo[['label','txt','txt_clean']]

  # Split train test
  train, test = model_selection.train_test_split(corpus, 
      test_size=0.2, stratify=corpus['label'], random_state=rand_state)

  # Split train validate
  train, valid = model_selection.train_test_split(train, 
      test_size=0.25, stratify=train['label'], random_state=rand_state)

  print(f"    train={train.shape}, valid={valid.shape}," +\
        f" test={test.shape}")

  return [train, valid, test]


def get_hyperparameters(params):
  ''' Return a list with hyperparameters based on the passed dictionary
  Adopted from:
    https://stackoverflow.com/questions/38721847/how-to-generate-all-combination-from-values-in-dict-of-lists-in-python
  Args:
    param (dict): a dictionary specified in the config.txt file.
  Return:
    param_list (list): a nested list of hyperparameters in the order of
      max_feature, ngram_range, and p_threshold
  '''

  keys, values = zip(*params.items())
  param_list = [v for v in itertools.product(*values)]
  
  return keys, param_list


def save_train_corpus_texts(train):
  # Write training texts to a folder where each file has 5000 entries.
  corpus_train_path = work_dir / "corpus_train"
  corpus_train_path.mkdir(parents=True, exist_ok=True)
  print("  save in:", corpus_train_path)

  # Note that I use the original text for training tokenizer
  txts  = train['txt'].values

  # list of training corpus files
  files = []             
  for idx in range(0,len(txts),5000):
    subset = txts[idx:idx+5000]
    subset_file = corpus_train_path / f"txt_{idx}"

    # force posix path to be string, otherwize the training step below will fail
    files.append(str(subset_file))
    with open(subset_file, "w") as f:
      subset_txts = '\n'.join(subset)
      f.write(subset_txts)

  return files


def train_tokenzier(files):
  # Intialize and train tokenizer
  special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]
  tokenizer = BertWordPieceTokenizer()
  tokenizer.train(files=files, vocab_size=vocab_size, 
                  min_frequency=min_frequency, special_tokens=special_tokens)
  
  # enable truncation up to the maximum 512 tokens
  tokenizer.enable_truncation(max_length=max_length)

  # save tokenizer
  model_path = work_dir / "model_bert"
  model_path.mkdir(parents=True, exist_ok=True)
  tokenizer.save_model(str(model_path))

  # dumping some of the tokenizer config to config file, 
  # including special tokens, whether to lower case and the maximum sequence length
  with open(model_path / "config.json", "w") as f:
      tokenizer_cfg = {
          "do_lower_case": True,
          "unk_token": "[UNK]",
          "sep_token": "[SEP]",
          "pad_token": "[PAD]",
          "cls_token": "[CLS]",
          "mask_token": "[MASK]",
          "model_max_length": max_length,
          "max_len": max_length,}
      json.dump(tokenizer_cfg, f)

  print("  tokenizer saved in:", model_path)

  # Load the tokenizer so it is callable
  tokenizer_loaded = transformers.BertTokenizerFast.from_pretrained(model_path)

  return tokenizer_loaded


def get_bert_model():

  # Set up input layers
  idx   = tf.keras.layers.Input((max_length), dtype="int32", name="input_idx")
  masks = tf.keras.layers.Input((max_length), dtype="int32", name="input_masks")

  ## get pre-trained bert with config
  config = transformers.DistilBertConfig(dropout=0.2, attention_dropout=0.2)
  config.output_hidden_states = False
  distbt = transformers.TFDistilBertModel.from_pretrained(
                                      'distilbert-base-uncased', config=config)  

  ## get the output tensors from the loaded bert model
  bert_out = distbt(idx, attention_mask=masks)[0]

  ## set up additional layers
  x     = tf.keras.layers.GlobalAveragePooling1D()(bert_out)
  x     = tf.keras.layers.Dense(64, activation="relu")(x)
  y_out = tf.keras.layers.Dense(2, activation='softmax')(x)

  ## put the layers together
  model = tf.keras.models.Model([idx, masks], y_out)
  for layer in model.layers[:3]:
      layer.trainable = False

  ## compile model
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
  loss      ='sparse_categorical_crossentropy'
  metrics   =['accuracy']
  model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

  return model


# Define function to encode text data in batches
def batch_encode(tokenizer, texts, batch_size=256):
  """""""""
  A function that encodes a batch of texts and returns the texts'
  corresponding encodings and attention masks that are ready to be fed 
  into a pre-trained transformer model.
  
  Input:
  - tokenizer:   Tokenizer object from the PreTrainedTokenizer Class
  - texts:       List of strings where each string represents a text
  - batch_size:  Integer controlling number of texts in a batch
  - max_length:  Integer controlling max number of words to tokenize in a
    given text
  Output:
  - input_ids:       sequence of texts encoded as a tf.Tensor object
  - attention_mask: the texts' attention mask encoded as a tf.Tensor obj
  """""""""
  # Define the maximum number of words to tokenize (up to 512)
  max_length = 512
  input_ids = []
  attention_mask = []
  
  for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    inputs = tokenizer.batch_encode_plus(batch,
                                          max_length=max_length,
                                          padding='max_length',
                                          truncation=True,
                                          return_attention_mask=True,
                                          return_token_type_ids=False
                                          )
    input_ids.extend(inputs['input_ids'])
    attention_mask.extend(inputs['attention_mask'])
  
  return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)


def encode_corpus(train, valid, test):

  # Convert dataframes to Datasets
  dataset_train = Dataset.from_pandas(train)
  dataset_valid = Dataset.from_pandas(valid)
  dataset_test  = Dataset.from_pandas(test)

  # The pre-processed text data is used for encoding purpose. From dataset data
  # type, the returned object from here are lists.
  X_train = dataset_train['txt_clean']
  X_valid = dataset_valid['txt_clean']
  X_test  = dataset_test['txt_clean']

  # Set up labels: cannot get these from dataset type since list cannot be used
  # to store labels for the model.fit function below. Instead, get them from
  # the original dataframe
  y_train = train['label']
  y_valid = valid['label']
  y_test  = test['label']

  # Encode corpus
  X_train_ids, X_train_attn = batch_encode(tokenizer_loaded, X_train)
  X_valid_ids, X_valid_attn = batch_encode(tokenizer_loaded, X_valid)
  X_test_ids , X_test_attn  = batch_encode(tokenizer_loaded, X_test)

  # Compile features for the input layers
  X_train_bert = [np.asarray(X_train_ids, dtype='int32'),
                  np.asarray(X_train_attn, dtype='int32')]
  X_valid_bert = [np.asarray(X_valid_ids, dtype='int32'),
                  np.asarray(X_valid_attn, dtype='int32')]
  X_test_bert  = [np.asarray(X_test_ids, dtype='int32'),
                  np.asarray(X_test_attn, dtype='int32')]

  return [X_train_bert, X_valid_bert, X_test_bert, y_train, y_valid, y_test]


def run_main_function():

  # Split train/validate/test for cleaned text
  #   Will not focus on original due to issues with non-alphanumeric characters
  #   and stop words.
  print("\nRead file and split train/validate/test...")
  [train, valid, test] = split_train_validate_test(corpus_combo_file, rand_state)

  print("\nSave training corpis as texts...")
  files = save_train_corpus_texts(train)

  print("\nTrain, save, and reload tokenizer...")
  tokenizer_loaded = train_tokenizer(files)

  print("\nCompile bert model...")
  model = get_bert_model()

  print("\nEncode train, valid, and test sets...")
  [X_train_bert, X_valid_bert, X_test_bert, y_train, y_valid, y_test] = \
      encode_corpus(train, valid, test)

  # get bert parameter list
  param_keys, param_list  = get_hyperparameters(bert_param)

  #### DID NOT FINISH THIS. BERT DID NOT PERFORM AS WELL. SO KIND OF A MOOT
  #### POINT TO DO TUNING.
  '''
  # iterate through different parameters
  with open(work_dir / f"scores_cln_bert", "w") as f:
    f.write("run\ttxt_flag\tlang_model\tparameters\tvalidate_f1\t" +\
            "test_f1\tmodel_dir\n")
    run_num = 0
    for param in param_list:
      print(f"\n## param: {param}")
      best_score, model_dir, test_score = run_pipeline(param, subsets)

      f.write(f"{run_num}\tcln\t{lang_model}\t{str(param)}\t"+\
              f"{best_score}\t{test_score}\t{model_dir}\n")

      run_num += 1


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

'''
