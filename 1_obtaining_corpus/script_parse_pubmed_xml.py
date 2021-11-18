################################################################################
#
# By:   Shin-Han Shiu
# Created: 11/15/21
# For: Screen through the pubmed baseline files and filter them to get plant
#   science records.
# 
# 11/16/21: 
# - Found that texts would be trucated if there is > inside. Fixed with new
#   get_value().
# - Abstract do not always start with <AbstractText>. Some are multiple parts
#   with different tags. But all are enclosed within <Abstract> and </Abstract>.
#   Fixed.
# 11/17/21:
# - Try to make the program more efficient by moving most elif into an elif in
#   the while loop of parse_xml(), but it is not really saving much time and is
#   buggy. Revert back to just a linear if-elif.   
################################################################################

import argparse, gzip, sys
from pathlib import Path
from os import listdir

#
# For: Iterate through xml files in the target directory
# Parameters: 
#   xmls_path - The Path object to the baseline xml files from NCBI Pubmed.
#   pnames - plant names for matching later.
# Output: 
#   A tab delimited file cotanining [PMID, Date, Journal, Title, Abstract]. 
#   The file has a .parse_tsv extension. One output is generated for each 
#   gzipped pubmed XML file.
#
def iterate_xmls(xmls_path, pnames):
    out_log = open(xmls_path / "log_error", "w")
    xmls = listdir(xmls_path)
    
    # Go through compressed xml files
    total_q = 0
    out_log2 = open("log_qualified", "w")
    out_log2.write("Pubmed\t#_qualified\n")
    for xml in xmls:
        if xml.endswith(".xml.gz"):
            print(xml)
            out_log.write(f'{xml}\n')
            xml_path = xmls_path / xml
            pubmed_d = parse_xml(xml_path, pnames, out_log)
            # output for each xml file
            out_path = xmls_path / (xml + ".parsed_tsv")
            doc_q    = 0
            with open(out_path, "w") as f:
                f.write("PMID\tDate\tJournal\tTitle\tAbstract\tQualifiedName\n")
                for ID in pubmed_d:
                    [TI, JO, AB, YR, MO, DY, Q] = pubmed_d[ID]
                    if Q != "":
                        f.write(f"{ID}\t{YR}-{MO}-{DY}\t{JO}\t{TI}\t{AB}\t{Q}\n")
                        doc_q += 1
            out_log2.write(f'{xml}\t{doc_q}')
            total_q += doc_q
    
    print("Total qualified:", total_q)
    out_log.close()
    out_log2.close()
    
#
# For: Go through an XML file and return a dictionary with PMID, title, date,
#   abstract, and journal name.
# Parameters: 
#   xml_gz_path - The Path object to an xml baseline gzipped file
#   pnames - plant names for matching later.
#   out_log - path to the log file for documenting errors.
# Return: 
#   pubmed_d - {pmid:[TI, JO, AB, YR, MO, DY, Qualified]}. Qualified indicates
#     if the record contains plant science related keyword.
#
def parse_xml(xml_path, pnames, out_log, debug=0):

    # Tags: 
    # 11/17/21: In the following order of appearance. Did not followed this
    #   and resulted in problems in the while loop below.
    PM = "<PMID Version="
    #PM = "ArticleId IdType=\"pubmed\""
    # 11/15/21: The above is no good because citation field also used the same
    #   format.
    AR = "<PubmedArticle>"    # new article
    TI = "<ArticleTitle>"     # title begin tag
    # 11/17/21: Realize that a small number of journals do not ISOAbbreviation.
    #   get full title instead.
    #JO = "<ISOAbbreviation>"
    JO = "<Journal>"
    JOe= "<Title>"
    AB = "<Abstract>"
    ABe= "</Abstract>"
    DA = "<PubMedPubDate PubStatus=\"pubmed\">"
    DAe= "</PubMedPubDate>"   # Note that other PubStatus also has the
                              # same end tag.
    YR = "<Year>"
    MO = "<Month>"
    DY = "<Day>"
    
    # {pmid:[TI, JO, AB, YR, MO, DY, Qualified]}
    pubmed_d = {}
    
    # read file line by line
    input_obj = gzip.open(xml_path, 'rt') # read in text mode
    L         = input_obj.readline()
    c         = 0
    PMID      = ""
    ABSTRACT  = ""
    # PMID already found or not
    flag_PMID = 0
   
    # whether DA tag is found or not before encoutering an DA end tag.
    flag_DA   = 0
    # whether AB tag is found or not.
    flag_AB   = 0
    
    # Title or Abstract contains plant-related keywords
    while L != "":
        L = L.strip()
        if debug:
            print([L])
        if L.startswith(AR) != 0:
            # This is set to make sure PMID tag is found for this article
            # and values are stored in the dictionary.
            flag_PMID = 0
            if debug:
                print("New record")
            
            if c % 1e3 == 0:
                print(f' {c/1e3} x 1000')
            c+= 1
        # 11/16/21: Found that the same PMID tag can occur multiple times for
        #   the same XML record, so use a flag to control for it.
        elif L.startswith(PM) and flag_PMID == 0:
            PMID = get_value(L)
            flag_PMID = 1
            if debug:
                print("-->",PMID)
            if PMID not in pubmed_d:
                pubmed_d[PMID] = ["","","","","","",""]
            else:
                out_log.write(f" Redun:{PMID}\n")
        # 11/17/21: Some records don't have titles or abstracts. Ignore.
        #   e.g., 31722833.
        elif L.startswith(TI):    # Title
            pubmed_d[PMID][0] = get_value(L)
            if debug:
                print("--> TI:",pubmed_d[PMID][0][:30],"...")
        elif L == JO:    # Journal 
            flag_JO = 1
            if debug:
                print("--> JO start tag")
            # have an inner loop till the end tag is found
            L = input_obj.readline()
            while L != "":
                L = L.strip()
                if L.startswith(JOe):
                    pubmed_d[PMID][1] = get_value(L)
                    if debug:
                        print("--> JO:", pubmed_d[PMID][1])
                    break
                L = input_obj.readline()
            
        elif L == AB:             # Abstract
            flag_AB = 1
            if debug:
                print("--> AB start tag")
        # Populate abstract text if encountering the beginning abstract tag
        elif flag_AB and L != ABe:
            ABSTRACT += L            
            if debug:
                print("--> AB populate")
        elif L == ABe and flag_AB == 1:
            pubmed_d[PMID][2] = get_value(ABSTRACT)
            flag_AB = 0
            ABSTRACT = ""
            if debug:
                print("--> AB end tag")
                print("--> AB:",pubmed_d[PMID][2][:30],"...")
        # Deal with date
        elif L.startswith(DA):
            if debug:
                print("--> DA start tag")
            flag_DA = 1
        elif L.startswith(YR) and flag_DA == 1:
            pubmed_d[PMID][3] = get_value(L)
            if debug:
                print("--> YR:",pubmed_d[PMID][3])
        elif L.startswith(MO) and flag_DA == 1:
            pubmed_d[PMID][4] = get_value(L)
            if len(pubmed_d[PMID][4]) == 1:
                pubmed_d[PMID][4] = "0" + pubmed_d[PMID][4]
            if debug:
                print("--> MO:",pubmed_d[PMID][4])
        elif L.startswith(DY) and flag_DA == 1:
            pubmed_d[PMID][5] = get_value(L)
            if len(pubmed_d[PMID][5]) == 1:
                pubmed_d[PMID][5] = "0" + pubmed_d[PMID][5]        
            if debug:
                print("--> MO:",pubmed_d[PMID][5])
        # Encouter Date end tag when a corresponding begin tag exists. This is
        # the end of the record for this entry. Reset values
        elif L.startswith(DAe) and flag_DA == 1:
            flag_DA = 0
            if debug:
                print("--> DA end tag")
                print("[Done]",pubmed_d[PMID],"\n")   

            # See if this is a plant science related record
            try:
                pubmed_d[PMID][6] = match_plant_names(pubmed_d[PMID][0], 
                                                      pubmed_d[PMID][2], 
                                                      pnames)
            except IndexError:
                print("IndexErr:",pubmed_d[PMID])
                sys.exit(0)
                
        L = input_obj.readline()
               
    print("  # articles:",c)
    out_log.write(f" # articles:{c}")
    
    return pubmed_d

# Rid of the tags
def get_value(L):
    # Get the 1st chuck of text 
    tag1_L_b = L.find("<")  # beginning html tag, Left, beginning
    tag1_L_e = L.find(">")  # beginning html tag, Left, ending
    tag1_R_b = L.find("</") # beginning html tag, Right, beginning
    
    # Text example
    # L1           L2      L3
    # blah blah <i>bleh</i> and <d>blue</d>.
    # Also work if text starts with tag.
    L1 = L[:tag1_L_b]
    L2 = L[tag1_L_e+1 : tag1_R_b]
    L3 = L[tag1_R_b:]
    tag1_R_e = L3.find(">") # beginning html tag, Right, ending
    L3 = L3[tag1_R_e+1:]
    
    L = L1 + L2 + L3
    
    # Check if there is more tag, if so, run the function again
    if len(L.split("<")) > 1:
        L = get_value(L)
        
    return L

# For: Read names for the name files with both NCBI taxnomy and USDA entries.
# Parameters:
#   pnames_path - path to the plant name file.
# Return:
#   pnames - If a pname have multiple tokens, {token:{pname:0}}. If just one,
#     then {pname:1}. This is to take care of compound plant names.
def read_plant_names(pnames_path):
    pnames = {}
    with open(pnames_path) as f:
        # Note: match lower case
        L = f.readline().strip().lower()
        while L != "":
            if L[0] != "#":
                tokens = L.split(" ")
                # This will ensure that, even if pname[L] has already been
                # assigned a dictionary as value because there is >=1 plant
                # names with L as a part, the simpler name L will override the
                # more complicated names.
                if len(tokens) == 1:
                    pnames[L] = 1
                else:
                    for token in tokens:
                        # Take care of situation where a plant name (e.g. moss)
                        # is also part of a compound plant name. In that case,
                        # only the simple plant name will be stored.
                        if token in pnames:
                            if pnames[token] == 1:
                                continue
                            else:
                                pnames[token][L] = 0
                        else:
                            pnames[token] = {L:0}
                            
            L = f.readline().strip()
    return pnames
    
def match_plant_names(TI, AB, pnames):
    
    # Look at title first
    Qualified = match_function(TI, pnames)
    
    # Look at abstract if no match in title
    if not Qualified:
        Qualified = match_function(AB, pnames)
        
    return Qualified

def match_function(doc, pnames, debug=0):
    Qualified = ""
    if debug:
        print(f"Check:", doc[:20],"...")
    doc = doc.strip().lower()
    for token in doc.split(" "):
        token = token.strip()
        if token in pnames:
            #if debug == 1:
            #    print("--|",token)
                
            # Take care of compound plant names
            # This is not a compound plant name, so no need to do anything.
            if pnames[token] == 1:
                if debug == 1:
                    print(" -----> Match1:", token)
                Qualified = token
            # A compound plant name, need to do find for the entire corpus.
            else:
                for compound_name in pnames[token]:
                    if doc.find(compound_name) != -1:
                        if debug == 1:
                            print(" -----> Match2:", compound_name)
                        Qualified = compound_name
                        break
            if Qualified != "":
                break
    return Qualified
    
################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('-x', '--xmls_path', help='path to gzipped xmls',
                    required=True)
parser.add_argument('-p', '--pnames_path', help='path to plant name file',
                    required=True)
args = parser.parse_args()

xmls_path = Path(args.xmls_path)
pnames_path = Path(args.pnames_path)

# Get plant names
pnames = read_plant_names(pnames_path)

iterate_xmls(xmls_path, pnames)
