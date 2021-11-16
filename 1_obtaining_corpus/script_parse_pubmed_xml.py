################################################################################
#
# By:   Shin-Han Shiu
# Created: 11/15/21
# For: Screen through the pubmed baseline files and filter them to get plant
#   science records.
# 
# 11/16/21: Found that texts would be trucated if there is > inside.
################################################################################

import argparse
import spacy
import gzip
from spacy.matcher import Matcher
from pathlib import Path
from os import listdir

#
# For: Iterate through xml files in the target directory
# Parameters: 
#   xmls_path - The Path object to the baseline xml files from NCBI Pubmed.
# Return: 
#
def iterate_xmls(xmls_path):
    out_log = open(xmls_path / "log_iterate_xmls", "w")
    out_pub = open(xmls_path / "parse_records", "w")
    out_pub.write("PMID\tDate\tJournal\tTitle\tAbstract\n")
    xmls = listdir(xmls_path)
    
    # Go through compressed xml files
    for xml in xmls:
        if xml.endswith(".xml.gz"):
            print(xml)
            out_log.write(f'{xml}\n')
            xml_path = xmls_path / xml
            pubmed_d = parse_xml(xml_path, out_log)
            for ID in pubmed_d:
                [TI, AB, JO, YR, MO, DY] = pubmed_d[ID]
                out_pub.write(f"{ID}\t{YR}-{MO}-{DY}\t{JO}\t{TI}\t{AB}\n")
    
    out_log.close()
    out_pub.close()

#
# For: Go through an XML file and return a dictionary with PMID, title, date,
#   abstract, and journal name.
# Parameters: 
#   xml_gz_path - The Path object to an xml baseline gzipped file
# Return: 
#   pubmed_d - {pmid:[TI, AB, JO, YR, MO, DY]}
#
def parse_xml(xml_path, out_log):

    # Tags 
    AR = "PubmedArticle"    # new article
    TI = "ArticleTitle"     # title begin tag
    AB = "AbstractText"
    JO = "ISOAbbreviation"
    DA = "PubMedPubDate PubStatus=\"pubmed\""
    DAe= "/PubMedPubDate"   # Note that other PubStatus also has the
                            # same end tag.
    YR = "Year"
    MO = "Month"
    DY = "Day"
    #PM = "ArticleId IdType=\"pubmed\""
    # The above is no good because citation field also used the same format.
    PM = "PMID Version="

    # {pmid:[TI, AB, JO, YR, MO, DY]}
    pubmed_d = {}
    
    # read file line by line
    input_obj = gzip.open(xml_path, 'rt') # read in text mode
    L         = input_obj.readline()
    c         = 0
    fields    = ["","","","","",""] # [TI, AB, JO, YR, MO, DY] 

    # whether DA_b flag is found or not before encoutering an DA end tag.
    flag_DA   = 0
    PMID      = ""
    while L != "":
        L = L.strip()
        if L.startswith(f"<{AR}>") != 0:
            # This is set to make sure PMID tag is found for this article
            # and values are stored in the dictionary.
            fields   = ["","","","","",""]
            PMID = ""
            #print("\n",L)parse_args
            if c % 1e4 == 0:
                print(f'   {c}')
            c+= 1
        elif L.startswith(f"<{TI}") != 0:
            fields[0] = get_value(L)
        elif L.startswith(f"<{AB}") != 0:
            fields[1] = get_value(L) 
        elif L.startswith(f"<{JO}") != 0:
            fields[2] = get_value(L)
        elif L.startswith(f"<{DA}") != 0:
            flag_DA = 1
        elif L.startswith(f"<{YR}") != 0 and flag_DA == 1:
            fields[3] = get_value(L)
        elif L.startswith(f"<{MO}") != 0 and flag_DA == 1:
            fields[4] = get_value(L)
            if len(fields[4]) == 1:
                fields[4] = "0" + fields[4]
        elif L.startswith(f"<{DY}") != 0 and flag_DA == 1:
            fields[5] = get_value(L)
            if len(fields[5]) == 1:
                fields[5] = "0" + fields[5]        
        # Encouter Date end tag when a corresponding begin tag exists
        elif L.startswith(f"<{DAe}") != 0 and flag_DA == 1:
            flag_DA = 0
        elif L.startswith(f"<{PM}") != 0:
            PMID = get_value(L)
            if PMID not in pubmed_d:
                pubmed_d[PMID] = fields
            else:
                out_log.write(f" Redun:{PMID}\n")
            #print("PMID:",PMID)
            #print(fields)
        L = input_obj.readline()
               
    print("  # articles:",c)
    out_log.write(f" # articles:{c}")
    
    return pubmed_d

# Rid of the tags
def get_value(L):
    L = L.split(">")[1]
    L = L.split("</")[0]
    return L
    
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('-x', '--xmls_path', help='path to gzipped xmls')
args = parser.parse_args()

xmls_path = Path(args.xmls_path)
iterate_xmls(xmls_path)
