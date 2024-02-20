#
# Author: Shin-Han Shiu
# Date: 10/20/23
# Purpose: search for species information based on a dataset with time and
#   corpus information.
# 

####
# Setup
####
import pickle, nltk, re, multiprocessing, argparse, yaml, math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.sparse import csr_matrix, vstack
from time import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
from collections import OrderedDict, Counter
from bisect import bisect
from mlxtend.preprocessing import minmax_scaling
from copy import deepcopy

# needed data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# So PDF is saved in a format properly
mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams["font.family"] = "sans-serif"

####
# Functions
####
def parse_arguments():
  '''Get arguments'''
  parser = argparse.ArgumentParser()

  parser.add_argument("-c", "--config_file", type=str, required=True, 
    help="Configuration file in yaml format",
  )
 
  return parser.parse_args()

def get_name_dict(config):
  '''Get the tax_id of target taxa and generate a dictionary.
  Args:
    config (dict): Configuration dictionary
  Return:
    target_id - The NCBI taxon ID for the taxon.
    names_dict - A dictionary with: {tax_id:{name_class:[names]}
  '''

  target = config['set_taxa']['base_taxa']
  names_dmp_path = aux_dir / config['aux_data']['names_dmp_path']
  names_dict_path = work_dir / f"{target}_taxa_names_dict.pickle"
  target_id_path  = work_dir / f"{target}_target_id.txt"

  if names_dict_path.is_file() and target_id_path.is_file():
    print("  load names_dmp and target_id")
    with open(names_dict_path, "rb") as f:
      names_dict = pickle.load(f)
    with open(target_id_path, "r") as f:
      target_id = f.readline().strip()
  else:
    print("  parse names_dmp")
    target_id = ""
    names_dmp = open(names_dmp_path)
    L         = names_dmp.readline()
    names_dict = {}
    while L != "":
      L = L.strip().split("\t")
      tax_id = L[0]
      name   = L[2]
      name_c = L[6]
      if L[2] == target:
        #print(f"{target} tax_id:",tax_id)
        target_id = tax_id

      if tax_id not in names_dict:
        names_dict[tax_id] = {name_c:[name]}
      elif name_c not in names_dict[tax_id]:
        names_dict[tax_id][name_c] = [name]
      else:
        names_dict[tax_id][name_c].append(name)
      L = names_dmp.readline()
    
    with open(names_dict_path, "wb") as f:
      pickle.dump(names_dict, f)
    with open(target_id_path, "w") as f:
      f.write(target_id)

  print(f"    target: {target}, target_id: {target_id}")

  return target_id, names_dict

def get_parent_child(config):
  '''Get the parent-child relationships from nodes.dmp file.
  Args:
    config (dict): Configuration dictionary
  Return:
    parent_child - A dictionary with {parent:[children]}
    child_parent - A dictionary with {child:parent}
    rank_d       - A dictionary with {rank:count}
    taxa_rank    - A dictionary with {taxa_id:rank}
    rank_taxa    - A dictionary with {rank:[taxa_id]}
    debug_list   - A list of taxa_id that are children of base_taxa
  '''

  # The Path to the nodes.dmp file from NCBI taxonomy
  nodes_dmp_path = aux_dir / config['aux_data']['nodes_dmp_path']
  nodes_dmp_output = work_dir / "nodes_dmp_output.pickle"

  if nodes_dmp_output.is_file():
    print("  load nodes_dmp")
    with open(nodes_dmp_output, "rb") as f:
      [parent_child, child_parent, rank_d, taxa_rank, rank_taxa, debug_list] = \
                                                      pickle.load(f)
  else:
    print("  parse nodes_dmp")
    nodes_dmp  = open(nodes_dmp_path)
    L      = nodes_dmp.readline()
    rank_d     = {} # {rank: count}
    taxa_rank  = {} # {taxa_id: rank}
    rank_taxa  = {} # {rank: taxa_id}
    parent_child = {}
    child_parent = {}
    target_ranks = ['genus', 'family', 'order']

    debug_count  = 0
    debug_list   = []
    while L != "":
      L = L.strip().split("\t")
      tax_id = L[0]
      par_id = L[2]
      rank   = L[4]
      if rank not in rank_d:
        rank_d[rank] = 1
      else:
        rank_d[rank]+= 1
      
      # Don't want any species or taxon with no rank
      # 9/20/22: actually, do not want no rank result in problem. Am example
      #   is taxid=2822797, child of 147368, this lead to some taxa missing.
      #   so removed.
      #if rank not in ["no rank", "species"]:
      if rank != "species":
        # debug
        if par_id == '147383':
          debug_count += 1
          debug_list.append(
                  base_taxa_names_dict[tax_id]['scientific name'][0])
          #print(debug_count, tax_id, names_dict[tax_id]['scientific name'])

        # populate parent_child dict
        if par_id not in parent_child:
          parent_child[par_id] = [tax_id]
        else:
          parent_child[par_id].append(tax_id)
        
        # populate child_parent dict
        if tax_id not in child_parent:
          child_parent[tax_id] = par_id
        else:
          print(f"ERR: {tax_id} with >1 parents",
              child_parent[tax_id], par_id)
        
        # populate taxa_rank and rank_taxa dicts
        taxa_rank[tax_id] = rank
        
        if rank not in rank_taxa:
          rank_taxa[rank] = [tax_id]
        else:
          rank_taxa[rank].append(tax_id)
        
      L = nodes_dmp.readline()

    with open(nodes_dmp_output, "wb") as f:
      pickle.dump([parent_child, child_parent, rank_d, taxa_rank, rank_taxa, 
                   debug_list], f)
    
  return [parent_child, child_parent, rank_d, taxa_rank, rank_taxa, debug_list]

def get_offsprings(p, parent_child, offsprings, debug=0):
  '''Get the offsprings of a parent.
  Args:
    p - The parent taxa ID to get children for.
    paren_child - The dictionary returned from get_parent_child().
    offsprings - An initially empty list to append offspring IDs.
  Return:
    offsprings - The populated offspring list.
  '''
  if debug:
    print(p)

  if p in parent_child:
    # Initialize c with an empty element for debugging purpose
    #c = [""]
    c = parent_child[p]
    if debug:
      print("",p, c)
      if p == "147383":
        print("debug parent found")

    offsprings.extend(c)
    for a_c in c:
      get_offsprings(a_c, parent_child, offsprings)
  else:
    if debug:
      print(" NO CHILD")
  return offsprings

def check_duplicate(alist):
  '''Check if there is any duplicate name'''
  dup = [item for item, count in Counter(alist).items() if count > 1]
  print("duplicated:", dup)

def get_base_taxa_offspring(config):
  '''Get the offspring of base_taxa
  Args:
    config (dict): Configuration dictionary
  Return:
    base_taxa_offspr_names - A list of names of the offspring of base_taxa
  '''

  print("  get base_taxa offspring names")

  base_taxa    = config['set_taxa']['base_taxa']
  outfile_name = work_dir / f'{base_taxa}_offspring_names.pickle'
 
  if not outfile_name.is_file():
    base_taxa_offspr = get_offsprings(base_taxa_id, parent_child, [])
    # Convert taxa id into scientific names
    base_taxa_offspr_names = []
    redun = {}
    for o in base_taxa_offspr:
      if o in base_taxa_names_dict:
        for nc in base_taxa_names_dict[o]: # for each name_class
          if nc != 'authority': 
            for name in base_taxa_names_dict[o][nc]:
              if name not in redun:
                base_taxa_offspr_names.append(name)
                redun[name] = 0
              #else:
              #  print("Redun:", name)

    check_duplicate(base_taxa_offspr_names)

    # Save as pickle
    with open(outfile_name, "wb") as f:
      pickle.dump(base_taxa_offspr_names, f)
  else:
    print(f"    load {base_taxa}_base_taxa_offspr_names")
    with open(outfile_name, "rb") as f:
      base_taxa_offspr_names = pickle.load(f)

  # Note that this number is larger than offspring_33090 which contain indicies
  # This is because there are other names, like synonyms for each index.
  print("    num base_taxa_offspr_names:", len(base_taxa_offspr_names))

  return base_taxa_offspr_names

def get_usda_common_names(config, base_taxa_offspr_names):
  '''Get the USDA common names
  Args:
    config (dict): Configuration dictionary
    base_taxa_offspr_names - A list of names of the offspring of base_taxa
  Return:
    cnames: {common_name:[scientific name, family]}
    common_names - A list of names of the offspring of base_taxa that are in
  '''

  print("  get usda common names")
  usda_plant_db = aux_dir / config['aux_data']['usda_plant_db']

  # common name file
  cnames_file = work_dir / "usda_common_names_dict.pickle"

  if not cnames_file.is_file():
    cnames = {} # {common_name:[scientific name, family]}

    with open(usda_plant_db) as f:
      f.readline() # header, don't need it
      L = f.readline()
      while L != "":
        L = L.strip()
        # There is empty line in the file.
        if L == "":
          break
        #print(L.split(","))
        try:
          # some names have "," in there. So need to split with ""\,"
          [symbol, synonym, sname, cname, fam] = L.split("\",")
        except ValueError:
          print("ValueError:",[L])
          break
        # rid of quotes
        [symbol, synonym, sname, cname, fam] = [symbol.split("\"")[1], 
                                                synonym.split("\"")[1], 
                                                sname.split("\"")[1], 
                                                cname.split("\"")[1], 
                                                fam.split("\"")[1]]
        # Get genus name out
        genus = sname.split(" ")[0]
        if cname != "":
          if cname not in cnames:
            cnames[cname] = [genus, fam]
        L = f.readline()
    # Save as pickle
    with open(cnames_file, "wb") as f:
      pickle.dump(cnames, f)
  else:
    print("    load cnames")
    with open(cnames_file, "rb") as f:
      cnames = pickle.load(f)

  print("    num cnames:", len(cnames))

  # Check if all USDA genus names are found in NCBI
  # This helped identified issues with the parent_child script and missing data
  # due to the use of older NCBI taxa dump file. Currently, missings ones are 
  # fungal and are excluded.
  cnames_overlap = {}
  for cname in cnames:
    genus = cnames[cname][0]
    if genus in base_taxa_offspr_names:
      cnames_overlap[cname] = genus

  print("    num cnames overlapped with NCBI taxa:", len(cnames_overlap))

  common_names = list(cnames_overlap.keys())

  return cnames, common_names

# Function based on Mauro Di Pietro (2020):
#  https://towardsdatascience.com/text-classification-with-no-model-training-935fe0e42180
# For the purpose here, did not do lower-casing
def utils_preprocess_text(text, lst_stopwords, flg_stemm=False, flg_lemm=True):
  '''
  Preprocess a string.
  :parameter
    :param text: string - name of column containing text
    :param lst_stopwords: list - list of stopwords to remove
    :param flg_stemm: bool - whether stemming is to be applied
    :param flg_lemm: bool - whether lemmitisation is to be applied
  :return
    cleaned text
  '''
  ## clean: stripping, then removing punctuations.
  text = str(text).strip()
  
  # RE: replace any character that is not alphanumeric, underscore, whitespace
  #  with ''. Originally this is it, but realized that biological terms have
  #  special characters including roman numerals, dash, and ",". So they are
  #  not removed.
  #text = re.sub(r'[^\w\s(α-ωΑ-Ω)-,]', '', text)
  # Use the original method
  text = re.sub(r'[^\w\s]', '', text)

  ## Tokenize (convert from string to list)
  lst_text = text.split()  
  
  ## remove Stopwords
  if lst_stopwords is not None:
    lst_text = [word for word in lst_text if word not in 
          lst_stopwords]
        
  ## Stemming (remove -ing, -ly, ...)
  if flg_stemm == True:
    ps = nltk.stem.porter.PorterStemmer()
    lst_text = [ps.stem(word) for word in lst_text]
        
  ## Lemmatisation (convert the word into root word)
  if flg_lemm == True:
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    lst_text = [lem.lemmatize(word) for word in lst_text]
      
  ## back to string from list
  text = " ".join(lst_text)
  return text

def preprocess(config):
  '''Preprocess the text in the corpus'''

  corpus_file = aux_dir / config['aux_data']['corpus_file']

  corpus = pd.read_csv(corpus_file, compression='gzip')
  print("  corpus:", corpus.shape)

  txt_clean_file = work_dir / "txt_clean.pickle"

  if not txt_clean_file.is_file():
    tqdm.pandas(desc="  clean text")
    lst_stopwords = nltk.corpus.stopwords.words("english")
    txt_clean    = corpus["txt"].progress_apply(lambda x: 
                                      utils_preprocess_text(x, lst_stopwords))
    with open(txt_clean_file, "wb") as f:
      pickle.dump(txt_clean, f)              
  else:
    print("  load cleaned txt...")
    with open(txt_clean_file, "rb") as f:
      txt_clean = pickle.load(f)

  return corpus, txt_clean

def get_match_csr_batch(txt):
  '''Get the match matrix for a batch of docs
  Args:
    txt (pd.Series): The text to match
  Return:
    match_csr_batch (csr_matrix): The sparse matrix with shape=(num_docs, 
      num_names)
  '''
  with multiprocessing.Pool(processes=15) as pool:
    results_ncbi_list = list(tqdm(pool.imap(task, enumerate(txt), chunksize=1), 
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
  match_csr_batch = csr_matrix((csr_val, (row_idx, col_idx)),
                         shape=(txt.shape[0], len(base_taxa_offspr_names)), 
                         dtype=np.int0)

  return match_csr_batch

def task(item):
  '''Task to parallelize
  Args:
    item (tuple): (row_number, doc)
  Return:
    results_ncbi (list): an offspring_name is present in the doc (1) or not(1)
  '''
  (row, doc) = item

  # pad the doc so if qualified name is at the beginning or end will still match
  doc = f" {doc} " 
  # Get the matching common names as a list
  # Get lower case because the common name can be the 1st word.
  doc_lower = doc.lower()
  results_usda = [name for name in common_names if(f" {name} " in doc_lower)]

  # Add the results to doc
  for cname in results_usda:  # for each common name
    try:
      genus = cnames[cname][0]  # get the genus name
    except TypeError:
      print("ERR:, cname:", cname, "cnames[cname]:", cnames[cname])
    doc += f" {genus}"        # add the genus name to doc
  
  # Match to NCBI names
  results_ncbi = [1 if(f" {name} " in doc) else 0 \
                                          for name in base_taxa_offspr_names]

  return results_ncbi

def get_match_csr(config):
  '''Get the match matrix for the corpus
  Args:
    config (dict): Configuration dictionary
  Return:
    match_csr (csr_matrix): The sparse matrix with shape=(num_docs, num_names)
  '''

  match_csr_file = work_dir / "match_csr.pickle"

  if match_csr_file.is_file():
    print("  load match_csr")
    with open(match_csr_file, "rb") as f:
      match_csr = pickle.load(f)
  else:
    n_subset = config['n_subset']
    n_batch  = math.ceil(txt_clean.shape[0]/n_subset)

    print(f"  get match csr in {n_batch} batches of {n_subset}")
    # Send a subset docs at a time so no memtory issue
    csr_list = []
    for idx in range(0, txt_clean.shape[0], n_subset):
      print(f"    [{idx}, {idx+n_subset})")
      # get subset of docs
      txt        = txt_clean[idx:(idx+n_subset)]
      # get csr
      match_csr_batch  = get_match_csr_batch(txt)
      csr_list.append(match_csr_batch)

    # stack csr
    match_csr = vstack(csr_list)

    with open(work_dir / "match_csr.pickle", "wb") as f:
      pickle.dump(match_csr, f)

  print("    final csr:", match_csr.shape)

  return match_csr

def get_and_set_binned_timestamps(config):
  '''Get timestamps for bins of equal number of entries, then turn each
     timestamp into a binned timestamp
  Args:
    config (dict): Configuration dictionary
  Return:

  '''

  print("  generate timestamps for bins")

  dates = corpus['Date']

  # Turn all dates into timestamps 
  timestamps = []
  for date in dates:
    [yr, mo, da] = date.split('-') # year, month, day
    dt   = datetime(int(yr), int(mo), int(da))
    ts   = dt.timestamp()
    timestamps.append(ts)
  timestamps.sort()

  # bin size
  num_bins = config["time_bin"]["num_bins"]
  bin_size = int(len(timestamps)/num_bins)
  print(f"    num_bins={num_bins}, bin_size={bin_size}")

  # index values of every bin
  bin_idxs = [idx for idx in range(0, len(timestamps), bin_size)]

  # timestamp values at bin_idxs
  bin_timestamps = [timestamps[idx] for idx in bin_idxs]

  # Modify the last value to be the max timestamp value + 1. This is because
  # the bin_size is rounded down the last value be smaller than the max
  # timestamp values. Also, +1 to the max value, otherwise the last entries will
  # be in its own bin.
  max_timestamp      = max(timestamps) + 1
  bin_timestamps[-1] = max_timestamp

  # dates correspond to the different timestamp
  bin_dates = [datetime.fromtimestamp(ts) for ts in bin_timestamps]
  # Put idx, timestamp, and date into a dataframe and save it.
  bin_df = pd.DataFrame(list(zip(bin_idxs, bin_timestamps, bin_dates)),
            columns=['bin_start_idx', 'bin_start_timestamp', 'bin_start_date'])
  bin_df.to_csv(work_dir / "df_bin_timestamp_date.csv", index=False)
  
  # Assign new timestamps based on the bin timestamp values
  ts_in_bins = []
  for date in dates:
    [yr, mo, da] = date.split('-') # year, month, day
    dt   = datetime(int(yr), int(mo), int(da))
    ts   = dt.timestamp()

    bin_idx = bisect(bin_timestamps, ts)

    if bin_idx < len(bin_timestamps):
      ts2     = bin_timestamps[bin_idx]
    # Deal with the last bin
    else:
      ts2     = datetime(2022, 12, 31).timestamp()
    ts_in_bins.append(ts2) 

  print("    ", len(ts_in_bins), ts_in_bins[:2])

  return ts_in_bins

def get_docs_with_bin_timestamps(ts_in_bins):
  '''Generate a new dataframe with date, cleaned text, and bin info
  Args:
    ts_in_bins (list): A list of timestamps that are binned
  Return:
    documents (pd.DataFrame): A dataframe with Date, txt_clean, Timestamps, 
      Bins, and Bins_left
  '''

  # Create a new dataframe with PMID, date, txt_clean, bin, and bin_left
  documents = pd.DataFrame({"Date": corpus['Date'], 
                            "txt_clean": corpus['txt_clean'], 
                            "Timestamps":ts_in_bins})
  documents = documents.sort_values("Timestamps")

  # a list of tuples showing the bin range (+/-1 of the unique val)
  ts_bins   = [pd.Interval(left=ts-1, right=ts+1) for ts in ts_in_bins] 

  documents["Bins"]      = ts_bins
  documents["Bins_left"] = documents.apply(lambda row: row.Bins.left, 1)
  
  # Get unique bin timestamp values and sort them
  ts_unique = documents.Bins_left.unique()
  ts_unique.sort()

  return documents, ts_unique

def get_count_for_level(config):
  '''Get the number of taxa at the specified level in the base taxa'''

  target_level = config['set_taxa']['target_level']
  
  taxa_ids = rank_taxa["genus"]

  # convert tax_ids to scientific names, note that this contain ALL names at the
  # target level, not restricted to those in viridiplantae
  target_level_names_dict = {base_taxa_names_dict[tax_id]['scientific name'][0]:1
                      for tax_id in taxa_ids}
  target_level_names = list(target_level_names_dict.keys())

  # Get match_csr column index for taxa names at the target level:
  # e.g., there are 17512 genus names among Viridiplantae taxa

  # convert list to dict for more efficient operation
  offspr_dict = {name:1 for name in base_taxa_offspr_names}

  #https://stackoverflow.com/questions/50756085/how-to-print-the-progress-of-a-list-comprehension-in-python
  target_level_csr_idx = [base_taxa_offspr_names.index(name) \
                            for name in tqdm(target_level_names)
                            if name in offspr_dict]
  print(f"    num of {target_level}: {len(target_level_csr_idx)}")

  # Get the target level sub-csr
  target_level_csr = match_csr[:, target_level_csr_idx]
  print("    target_level_csr:", target_level_csr.shape)

  # Also compile the taxa names in the target level in the base taxa
  target_level_names_in_base = [name for name in tqdm(target_level_names)
                         if name in offspr_dict]
  
  # output match_csr for target level
  target_level_csr_file = f"match_csr_{target_level}.pickle"
  with open(work_dir / target_level_csr_file, "wb") as f:
    pickle.dump(target_level_csr, f)

  # output genus name in the same order as match_csr_genus
  base_taxa = config['set_taxa']['base_taxa']
  target_level_names_in_base_file = \
                      f"match_csr_{target_level}_names_in_{base_taxa}.pickle"
  with open(work_dir / target_level_names_in_base_file, "wb") as f:
    pickle.dump(target_level_names_in_base, f)

  # Total for each genus
  col_sum = np.squeeze(np.asarray(target_level_csr.sum(axis=0)))
  target_level_count_file = work_dir / f"{target_level}_count.txt"

  if not target_level_count_file.is_file():
    print("    load target_level_count_df")
    target_level_count_df = pd.DataFrame(
                                list(zip(target_level_names_in_base, col_sum)),
                                columns=("Taxa names", "Total"))
    target_level_count_df.to_csv(target_level_count_file)
  else:
    print("    read target_level_count_df")
    target_level_count_df = pd.read_csv(target_level_count_file, index_col=0)

  drop = config["set_taxa"]["drop"]
  target_level_count_df = \
    target_level_count_df[~target_level_count_df['Taxa names'].isin(drop)]

  return target_level_csr, target_level_count_df

def get_topX_ts_bin_counts(topX, match_csr, level_offspr, level_parent, count_df, 
                           base_taxa_offspr_names):
  '''Get topX taxa, then count the numbers of docs in each bin
  Args:
    topX (int): number of top taxa to look at
    match_csr (csr): csr with counts for the offspring taxa of interests
    level_offspr (str): taxonmic level of the offsprings to focus on, e.g. genus
    level_paraent (str): the parent level, e.g. viridiplantae. This is for
      naming the output file.
    count_df (DataFrame): two columns, "Taxa names" and "Total"
    base_taxa_offspr_names (list): a list of names of the level of interest that
      are also in the viridi_offspr_dict (with names of Viridiplantae offspring
      taxa).
  Return:
    ts_count_df(DataFrame): timestamp as rows, taxa as columns, counts as vals
    level_ts (DataFrame): taxa as rows, timestamp as columns, min-max
      normalized counts as values
  '''      
  # Get the sub-dataframe for topX
  count_df_topX = count_df.nlargest(topX, 'Total')
  #print(count_df_topX)
  
  topX_names    = count_df_topX['Taxa names'].tolist()
  #print(topX_names)
  
  # Get the csr indices for the topX names
  topX_idx = [idx for idx, name in enumerate(base_taxa_offspr_names)
                if name in topX_names]
  print("    topX:", len(topX_idx))
  #print(topX_idx)

  # Create a dict with {top_name:{timestamp:count}
  topX_ts_count = {}
  # the beginning ts of bins of all docs
  ts_all        = documents["Bins_left"] 

  # Go through each column. Only look at the topX
  for col_idx in tqdm(topX_idx):
    # Get column values
    col_val  = match_csr[:,col_idx].toarray().ravel()
    top_name = base_taxa_offspr_names[col_idx]
    topX_ts_count[top_name] = OrderedDict()
    for idx, ts in enumerate(ts_all):
      doc_val = col_val[idx]
      if ts not in topX_ts_count[top_name]:
        topX_ts_count[top_name][ts] = doc_val
      else:
        topX_ts_count[top_name][ts]+= doc_val
  
  # Convert to dataframe
  ts_count_df   = pd.DataFrame(topX_ts_count)

  # Sort df based on timestamps
  ts_count_df.sort_index(inplace=True)

  # Do min-max scaling
  ts_count_minmax_df   = minmax_scaling(ts_count_df,
                                              columns=ts_count_df.columns)
  # Add a date column
  ts_count_df['Date'] = ts_count_df.index.map(lambda ts: 
                                              datetime.fromtimestamp(ts))
  ts_count_minmax_df['Date'] = ts_count_df['Date']

  # Save dataframes
  ts_count_file = \
      f"{level_offspr}-of-{level_parent}_top{topX}_count_ts_bins.txt"
  ts_count_minmax_file = \
      f"{level_offspr}-of-{level_parent}_top{topX}_count_ts_bins_minmax.txt"
  ts_count_df.to_csv((work_dir / ts_count_file))
  ts_count_minmax_df.to_csv((work_dir / ts_count_minmax_file))
  
  # Transpose  ts_genus_count_minmax_df so the genus are in rows, ts in columns.
  level_ts = ts_count_minmax_df.transpose()

  return ts_count_df, level_ts

#-------------------------------------------------------------------------------
if __name__== '__main__':

  ###
  # Get arguments and configuration
  args = parse_arguments()
  with open(args.config_file, 'r') as file:
    config = yaml.safe_load(file)

  ###
  # Setup directories
  proj_dir = Path(config['proj_dir'])
  work_dir = proj_dir / config['work_dir']
  aux_dir  = work_dir / config['aux_data']['aux_dir']

  ###
  # Step 1
  print("Get plant names")

  ## Parse name_dmp
  base_taxa_id, base_taxa_names_dict = get_name_dict(config)

  ## Parse nodes_dmp
  [parent_child, child_parent, rank_d, taxa_rank, rank_taxa, debug_list] = \
                                                      get_parent_child(config)
  ## Get offspring of base_taxa
  base_taxa_offspr_names = get_base_taxa_offspring(config)

  ## Parse USDA common names
  cnames, common_names = get_usda_common_names(config, base_taxa_offspr_names)

  ###
  # Step 2
  print("Find name in corpus")

  ## Preprocessing
  corpus, txt_clean = preprocess(config)

  ## Get match_csr
  match_csr = get_match_csr(config)

  ###
  # Step 3
  print("Geneate time bins")

  # Get and set timestamps for bins of equal number of entries
  ts_in_bins = get_and_set_binned_timestamps(config)

  # Get documents with bin timestamps
  documents, ts_unique = get_docs_with_bin_timestamps(ts_in_bins)

  ###
  # Step 4
  print("Get counts based on specified level")

  # Get count for target level
  _, target_level_count_df = get_count_for_level(config)

  # Get time over time bins
  _, _ = get_topX_ts_bin_counts(config["set_taxa"]["topX"],
                                match_csr, 
                                config["set_taxa"]["target_level"],
                                config["set_taxa"]["base_taxa"],
                                target_level_count_df, 
                                base_taxa_offspr_names)

  print("Done!")