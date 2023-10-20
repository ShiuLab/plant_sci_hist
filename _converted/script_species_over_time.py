#
# Author: Shin-Han Shiu
# Date: 10/20/23
# Purpose: search for species information based on a dataset with time and
#   corpus information.
# 

####
# Setup
####
import pickle, nltk, re, multiprocessing, argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, dok_matrix
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

def parse_arguments():
  '''Get arguments'''
  parser = argparse.ArgumentParser()

  parser.add_argument("-s", "--seed", type=int, required=False, default=231020
    help="Random seed for reproducibility",
  )

  parser.add_argument("-s", "--seed", type=int, required=False, default=231020
    help="Random seed for reproducibility",
  )

    
  parser.parse_args()


#-------------------------------------------------------------------------------
if __name__== '__main__':
  args = parse_arguments()