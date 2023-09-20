print("Import modules...")
import sys, os, pickle, argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans

def get_submatrices(X_vec, labels, target_label):
  '''Get the tf-idf sparse martriices of entries in and out of a cluster
  Args:
    X_vec (scipy csr): Tf-idf sparse matrix with docs as rows and features as
      columns.
    labels (list): a list of cluster labels of all docs
    target_label (int): the cluster label to focus on.
  Return:
    x_vec_target (scipy csr): sparse matrix of docs in target cluster
    x_vec_nontar (scipy csr): sparse matrix of docs outside of target cluster
  '''
  
  target_list  = []
  nontar_list  = []

  # Populate target and non-target lists with indices
  for i in range(len(labels)):
    label = labels[i]
    if label == target_label:
      target_list.append(i)
    else:
      nontar_list.append(i)

  # convert to numpy array
  target_array = np.array(target_list)
  nontar_array = np.array(nontar_list)

  # Get the sparse matrix columns based on indices
  X_vec_target = X_vec[target_array, :]
  X_vec_nontar = X_vec[nontar_array, :]
  #print(f"  target:{X_vec_target.shape}, non-target:{X_vec_nontar.shape}")

  return X_vec_target, X_vec_nontar

def ks_test(label):
  '''Get Kolmogorov-Smirnov test results for a cluster with label as index
  Args:
    label (int): cluster index
  Return:
    dict_results (dict): feature index as key, a list of effect size, KS-test
      statistic, and p-value as value.
  '''
  X_vec_target, X_vec_nontar = get_submatrices(X_vec, labels_kmeans, label)
  num_feat = X_vec_target.shape[1]

  dict_results = {} # {feat_index:[effect size, stat, pval]}
  for feat_index in tqdm(range(num_feat)):
    target_array = X_vec_target[:, feat_index].toarray().flatten()
    nontar_array = X_vec_nontar[:, feat_index].toarray().flatten()

    target_median = np.median(target_array)
    nontar_median = np.median(nontar_array)
    effect_size   = target_median-nontar_median

    if target_median > nontar_median:
      result = ks_2samp(target_array, nontar_array)
      dict_results[feat_index] = [effect_size, result.statistic, result.pvalue]

  return dict_results


print("Set parameters")
# Reproducibility
seed = 20220609

# Setting working directory
proj_dir   = Path.home() / "projects/plant_sci_hist"
work_dir   = proj_dir / "3_key_term_temporal/3_3_cluster_analysis"
work_dir.mkdir(parents=True, exist_ok=True)

os.chdir(work_dir)

# plant science corpus
dir25       = proj_dir / "2_text_classify/2_5_predict_pubmed"
corpus_file = dir25 / "corpus_plant_421658.tsv.gz"

# qualified feature names
dir31          = proj_dir / "3_key_term_temporal/3_1_pubmed_vocab"
X_vec_file     = dir31 / "tfidf_sparse_matrix_4542"
feat_name_file = dir31 / "tfidf_feat_name_and_sum_4542"

# fitted clustering objs
dir32            = proj_dir / "3_key_term_temporal/3_2_tf_idf_clustering"
clus_kmeans_file = dir32 / 'clus_kmeans'
cluster_num      = 500

# Expect 2 arguments:
parser = argparse.ArgumentParser()
parser.add_argument("start", type=int, 
                    help='starting cluster number, inclusive')
parser.add_argument("end", type=int,
                    help='end cluster number, inclusive')
args = parser.parse_args()

print("Load clusters")
with open(clus_kmeans_file, "rb") as f:
  clus_kmeans = pickle.load(f)

labels_kmeans = clus_kmeans.labels_

dict_kmeans = {}
for i in range(len(labels_kmeans)):
  label = labels_kmeans[i]
  if label not in dict_kmeans:
    dict_kmeans[label] = [i]
  else:
    dict_kmeans[label].append(i)

print("Load tf-idf sparse matrix")
# Load sparse matrix from a pickle
with open(X_vec_file, 'rb') as f:
  X_vec = pickle.load(f)

print("Determine KS-test stat")
dict_results_list = []
for label in range(args.start, args.end+1):
	dict_results = ks_test(label)
	dict_results_list.append(dict_results)
	

print("Dump the results as a pickle")
pickle_file = work_dir / f"dict_results_list_kmeans_C{args.start}-{args.end}"
with open(pickle_file, 'wb') as f:
  pickle.dump(dict_results_list, f)







