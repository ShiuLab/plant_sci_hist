# Proj: Plant science conceptual history
# By  : Shin-Han Shiu
# On  : 3/8/21  
#
# Entrez database and UID:
# https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi
# https://www.ncbi.nlm.nih.gov/books/NBK25497/table/chapter2.T._entrez_unique_identifiers_ui/?report=objectonly
# 
# Entrez rettype, retmode for efetch
# https://www.ncbi.nlm.nih.gov/books/NBK25499/table/chapter4.T._valid_values_of__retmode_and/
#
# 11/24/21: Modified so it retrival records based on a passed term list. And 
#   rid of methods unrelated to getting abstracts.
#

import sys, getopt
#import entrezpy.conduit
import hashlib
import gzip
import re
import os
from pathlib import Path

def get_abs(terms):
    
    terms_path = Path(terms)
    terms = open(terms_path).readlines()
    
    for term in terms:
        term = term.strip()
        print("Retrieve:", term)
        c    = entrezpy.conduit.Conduit('shius@msu.edu')
        pipe = c.new_pipeline()     

        # Parameter explained:
        # https://www.ncbi.nlm.nih.gov/books/NBK25499/
        sid  = pipe.add_search({'db'  : 'taxonomy', 
                               'term': term})
        fid  = pipe.add_fetch({'retmode' : 'xml',
                               'retmax'  : '10'}, 
                               dependency=sid)
        rec  = c.run(pipe)
          
#-------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--term_file_path', 
                    help='Path to a file where each term occupies a line',
                    required=True)
                    
args = parser.parse_args()
get_abs(terms)