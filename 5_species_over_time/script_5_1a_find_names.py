import pickle, multiprocessing
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.sparse import csr_matrix

def get_match_csr(txt):
  print(txt.shape)
  with multiprocessing.Pool(processes=15) as pool:
    results_ncbi_list = list(tqdm(pool.imap(task, enumerate(txt)), 
                                  total=len(txt)))

  row_idx   = []
  col_idx   = []
  csr_val   = []
  for row, results_ncbi in enumerate(results_ncbi_list):
    non0_idx = np.nonzero(results_ncbi)[0].tolist()
    row_idx.extend([row]*len(non0_idx))
    col_idx.extend(non0_idx)
    csr_val.extend([1]*len(non0_idx))

  # create a sparse matrix with shape=(num_docs, num_names)
  match_csr = csr_matrix((csr_val, (row_idx, col_idx)),
                         shape=(txt.shape[0], len(offspring_names)), 
                         dtype=np.int0)

  return match_csr

def task(item):
  '''Task to parallelize
  Args:
    item (tuple): (row_number, doc)
  Return:
    results_ncbi (list): an offspring_name is present in the doc (1) or not(1)
  '''
  (row, doc) = item
  # Get the matching common names as a list
  results_usda = [name for name in common_names if(f" {name} " in doc)]

  # Add the results to doc
  for cname in results_usda:  # for each common name
    genus = cnames[cname][0]  # get the genus name
    doc += f" {genus}"        # add the genus name to doc
  
  # Match to NCBI names
  results_ncbi = [1 if(name in doc) else 0 for name in offspring_names]

  return results_ncbi

#---------------
proj_dir   = Path.home() / "projects/plant_sci_hist"
work_dir   = proj_dir / "5_species_over_time/"

print("Read saved objects...")
txt_clean  = pd.read_csv(work_dir / "txt_clean.csv", index_col=0)

with open(work_dir / "viridiplantae_offspring_names.pickle", "rb") as f:
  offspring_names = pickle.load(f)

# Save as pickle
with open(work_dir / "usda_common_names.pickle", "rb") as f:
  common_names = pickle.load(f)

with open(work_dir / "usda_common_names_dict.pickle", "rb") as f:
  cnames = pickle.load(f)

print("Get match_csr...")
match_csr  = get_match_csr(txt_clean['txt_clean'])

with open(work_dir / "match_csr.pickle", "wb") as f:
  pickle.dump(match_csr, f)