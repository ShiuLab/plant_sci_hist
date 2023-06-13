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
#   rid of methods unrelated to getting abstracts. Use Biopython:
#   https://biopython.org/DIST/docs/tutorial/Tutorial.html#sec143
#
# 11/29/21: The XML obtained is not formated properly (i.e., all records are in
#   a long text string). Download medline format instead.
#
import sys, argparse
from Bio import Entrez
from pathlib import Path

#
# Get abstract
#
def get_abs(qstr, email, output_name):
    
    Entrez.email = args.email
    batch_size   = 50
    
    print("Retrieve:", qstr,)
    qstr = format_qstr(qstr)
    out_xml      = open("pubmed-" + output_name, "w")
    
    handle = Entrez.esearch(db="pubmed", 
                            term=qstr,
                            retmax="100000")
    record = Entrez.read(handle)
    idlist = record["IdList"]
    epost  = Entrez.epost("pubmed", id=",".join(idlist))
    epxml  = Entrez.read(epost)
    webenv = epxml["WebEnv"]
    qkey   = epxml["QueryKey"]
    
    count = int(len(idlist))
    print(f"{count} records")
    for start in range(0, count, batch_size):
        end = min(count, start + batch_size)
        print(f"  {start}-{end}")
        fetch  = Entrez.efetch(
            db="pubmed",
            rettype="medline",
            retmode="text",
            retstart=start,
            retmax=batch_size,
            webenv=webenv,
            query_key=qkey)
        data = fetch.read()
        fetch.close()
        out_xml.write(data)            
    
    out_xml.close()

# A query should look like.
#  "Jasmonic acid"+OR+Jasmonate+OR+JA
# Here quotes will be replaced with %22.
def format_qstr(qstr):
    qstr = "%22".join(qstr.split("\""))
    return qstr
          
#-------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--query', 
    help='Query string with a format like \"Jasmonic acid\"+OR+Jasmonate+OR+JA',
    required=True)
parser.add_argument('-e', '--email', 
    help='Email of individual submitting the query',
    required=True)
parser.add_argument('-o', '--output_name', 
    help='Name for output file',
    required=True)    
args = parser.parse_args()

get_abs(args.query,
        args.email,
        args.output_name)