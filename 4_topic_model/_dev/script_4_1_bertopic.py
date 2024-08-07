
import os, re, pickle
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

# output
docs_clean_file  = work_dir / "corpus_plant_421658_proc_txt.pkl"
topic_model_file = work_dir / "topic_model"
topics_file      = work_dir / "topics.pickle"

if docs_clean_file.is_file():
  print("Load processed docs")
  with open(docs_clean_file, "rb") as f:
    docs_clean = pickle.load(f)
else:
  print("Read corpus and process docs")
  corpus_df  = pd.read_csv(corpus_file, sep='\t', compression='gzip')
  docs       = corpus_df['txt']
  stop_words = stopwords.words('english')
  stop_words_dict = {}
  for i in stop_words:
    stop_words_dict[i] = 1

  docs_clean = []
  for doc_idx in tqdm(range(len(docs))):
    doc = docs[doc_idx]
    docs_clean.append(clean_text(doc, stop_words_dict))
  print("  # cleaned docs:",len(docs_clean))
  with open(docs_clean_file, "wb") as f:
    pickle.dump(docs_clean, f)

print("Run bertopic")

topic_model = BERTopic(calculate_probabilities=False,
                       n_gram_range=(1,2),
                       min_topic_size=1000, 
                       nr_topics='auto',
                       embedding_model='allenai-specter',
                       verbose=True)

topics = topic_model.fit_transform(docs_clean)

print("Save models and topics")

topic_model.save(topic_model_file)
with open(topics_file, "wb") as f:
  pickle.dump(topics, f)