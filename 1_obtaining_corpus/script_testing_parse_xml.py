import gzip
from pathlib import Path 
wd = Path('/mnt/d/project/plant_sci_history/1_obtaining_corpus/')

def parse_xml(xml_path, out_log):

    # Tags 
    AR = "PubmedArticle"    # new article
    TI = "ArticleTitle"     # title begin tag
    AB = "Abstract"
    ABe= "/Abstract"
    JO = "ISOAbbreviation"
    DA = "PubMedPubDate PubStatus=\"pubmed\""
    DAe= "/PubMedPubDate"   # Note that other PubStatus also has the
                            # same end tag.
    YR = "Year"
    MO = "Month"
    DY = "Day"
    #PM = "ArticleId IdType=\"pubmed\""
    # 11/15/21: The above is no good because citation field also used the same
    #   format.
    PM = "PMID Version="

    # {pmid:[TI, AB, JO, YR, MO, DY]}
    pubmed_d = {}
    
    # read file line by line
    input_obj = gzip.open(xml_path, 'rt') # read in text mode
    L         = input_obj.readline()
    c         = 0
    fields    = ["","","","","",""] # [TI, AB, JO, YR, MO, DY] 

    # whether DA tag is found or not before encoutering an DA end tag.
    flag_DA   = 0
    # whether AB tag is found or not.
    flag_AB   = 0
    ABSTRACT  = ""
    PMID      = ""
    while L != "":
        L = L.strip()
        #print(L, f'<{AB}>', L==f'<{AB}>')
        if L.startswith(f"<{AR}>"):
            # This is set to make sure PMID tag is found for this article
            # and values are stored in the dictionary.
            fields   = ["","","","","",""]
            PMID = ""
            #print("\n",L)parse_args
            if c % 1e4 == 0:
                print(f'   {c}')
            c+= 1
        # Title
        elif L.startswith(f"<{TI}"):
            fields[0] = get_value(L)
        # Deal with abstract lines
        elif L == f'<{AB}>':
            flag_AB = 1
            #print("FOUND AB flag")
        elif L.startswith(f"<{ABe}") and flag_AB == 1:
            fields[1] = get_value(ABSTRACT)
            # Reset
            flag_AB = 0
            AB      = ""
        # Journal 
        elif L.startswith(f"<{JO}"):
            fields[2] = get_value(L)
        # Deal with date
        elif L.startswith(f"<{DA}"):
            flag_DA = 1
        elif L.startswith(f"<{YR}") and flag_DA == 1:
            fields[3] = get_value(L)
        elif L.startswith(f"<{MO}") and flag_DA == 1:
            fields[4] = get_value(L)
            if len(fields[4]) == 1:
                fields[4] = "0" + fields[4]
        elif L.startswith(f"<{DY}") and flag_DA == 1:
            fields[5] = get_value(L)
            if len(fields[5]) == 1:
                fields[5] = "0" + fields[5]        
        # Encouter Date end tag when a corresponding begin tag exists
        elif L.startswith(f"<{DAe}") and flag_DA == 1:
            flag_DA = 0
        elif L.startswith(f"<{PM}"):
            PMID = get_value(L)
            if PMID not in pubmed_d:
                pubmed_d[PMID] = fields
            else:
                out_log.write(f" Redun:{PMID}\n")
            #print("PMID:",PMID)
            #print(fields)
            
        # Populate abstract text if encountering the beginning abstract tag
        if flag_AB:
            print([L])
            ABSTRACT += L
            
        L = input_obj.readline()
               
    print("  # articles:",c)
    out_log.write(f" # articles:{c}")
    
    return pubmed_d

def get_value(L):
    # Get the 1st chuck of text 
    tag1_L_b = L.find("<")  # beginning html tag, Left, beginning
    tag1_L_e = L.find(">")  # beginning html tag, Left, ending
    tag1_R_b = L.find("</") # beginning html tag, Right, beginning
    
    # Text example
    # L1           L2      L3
    # blah blah <i>bleh</i> and <d>blue</d>.
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

test_xml = wd / "_test/test.xml.gz"
out_log = open("log_test","w")
pubmed_d = parse_xml(test_xml, out_log)
out_log.close()

for i in pubmed_d:
    print(i)
    print(pubmed_d[i])