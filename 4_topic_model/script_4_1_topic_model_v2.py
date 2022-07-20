
print("Import modules")

import re, pickle, argparse, os
import pandas as pd
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from bertopic import BERTopic

def clean_text(x, stop_words_dict):
    x = str(x)
    x = x.lower()
    # Replace any non-alphanumric characters of any length
    # Q: Not sure what the # character do.
    x = re.sub(r'#[A-Za-z0-9]*', ' ', x)
    # tokenize and rid of any token matching stop words
    tokens = word_tokenize(x)
    x = ' '.join([w for w in tokens if not w in stop_words_dict])
    return x

print("Set parameters")

# Reproducibility
seed = 20220609

# Setting working directory
proj_dir   = Path.home() / "projects/plant_sci_hist"
work_dir   = proj_dir / "4_topic_model/4_1_get_topics"
work_dir.mkdir(parents=True, exist_ok=True)

os.chdir(work_dir)

# plant science corpus
dir25       = proj_dir / "2_text_classify/2_5_predict_pubmed"
corpus_file = dir25 / "corpus_plant_421658.tsv.gz"

# qualified feature names
dir31          = proj_dir / "3_key_term_temporal/3_1_pubmed_vocab"
X_vec_file     = dir31 / "tfidf_sparse_matrix_4542"
feat_name_file = dir31 / "tfidf_feat_name_and_sum_4542"

# Get the model name
parser = argparse.ArgumentParser()
parser.add_argument('model', type=str, help="Huggingface BERT model to use")
parser.add_argument('--num_docs', type=int, help="Number of docs to use")
args = parser.parse_args()
model_name = args.model

model_name_mod = "-".join(model_name.split("/"))

# output
docs_clean_file  = work_dir / f"docs_clean.pickle"
topic_model_file = work_dir / f"topic_model_{model_name_mod}"
topics_file      = work_dir / f"topics_{model_name_mod}.pickle"

print("Preprocessing")

if docs_clean_file.is_file():
  print("  load processed docs")
  with open(docs_clean_file, "rb") as f:
    docs_clean = pickle.load(f)
else:
  print("  read corpus and process docs")
  corpus_df = pd.read_csv(corpus_file, sep='\t', compression='gzip')
  
  docs       = corpus_df['txt']
  stop_words = stopwords.words('english')
  stop_words_dict = {}
  for i in stop_words:
    stop_words_dict[i] = 1

  docs_clean = []
  for doc_idx in tqdm(range(len(docs))):
    doc = docs[doc_idx]
    docs_clean.append(clean_text(doc, stop_words_dict))
  with open(docs_clean_file, "wb") as f:
    pickle.dump(docs_clean, f)

# Use a specified number of docs
if args.num_docs != None:
	docs_clean = docs_clean[:args.num_docs]
print("  num_docs=", len(docs_clean))

print("Run bertopic")
print("  model=", model_name)

if topic_model_file.is_file() and topics_file.is_file:
  print("  model already generated")
else:
  #topic_model = BERTopic(calculate_probabilities=False,
  #                      n_gram_range=(1,2),
  #                      min_topic_size=200, 
  #                      nr_topics='auto',
  #                      embedding_model=model_name,
  #                      verbose=True)
  topic_model = BERTopic(calculate_probabilities=False,
                         n_gram_range=(1,2),
                         embedding_model=model_name,
                         verbose=True)

  topics = topic_model.fit_transform(docs_clean)

  print("Save models and topics")

  topic_model.save(topic_model_file)
  with open(topics_file, "wb") as f:
    pickle.dump(topics, f)
