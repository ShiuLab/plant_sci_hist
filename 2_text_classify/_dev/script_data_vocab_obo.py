'''
Processing Ontology obo file

PO source: http://www.obofoundry.org/ontology/po.html
GO source: http://geneontology.org/docs/download-ontology/
Owlready2: https://pypi.org/project/Owlready2/

In KG/_vocab/vocab_plant_ontology
'''

import os, sys, getopt
from pathlib import Path

def get_names(infile, outfile):
    """ Parse the PO OBO file to get names and synonyms. For synonym, it has to
    have the NARROW flag
    
    [Term]
    id: PO:0000002
    name: anther wall
    namespace: plant_anatomy
    alt_id: PO:0006445
    alt_id: PO:0006477
    def: "A microsporangium wall (PO:0025307) ...
    comment: Has an outer epidermis ...
    subset: Angiosperm
    subset: reference
    synonym: "pared de la antera (Spanish, exact)" EXACT Spanish [POC:Maria_Alejandra_Gandolfo]
    synonym: "Poaceae anther wall (narrow)" NARROW []
    synonym: "pollen sac wall (exact)" EXACT []
    synonym: "Zea anther wall (narrow)" NARROW []
    synonym: "葯壁 (Japanese, exact)" EXACT Japanese [NIG:Yukiko_Yamazaki]
    xref: PO_GIT:149
    xref: PO_GIT:298
    is_a: PO:0025307 ! microsporangium wall
    relationship: part_of PO:0009066 ! anther
    """

    # Read all lines into a list
    obo_lns = open(infile, encoding="utf-8").readlines()
    
    oup = open(outfile, "w", encoding="utf-8")
    for i in obo_lns:
    
        if i.startswith("name:"):
            i = i[i.find(":")+1:].strip()
            oup.write(f"{i}\n")
        elif i.startswith("synonym:"):
            i = i[i.find("\"")+1: i.rfind("\"")]
            
            # PO has this (...) thing inside synonym:
            if i.find("(") != -1:
                i = i[:i.find("(")-1]
            oup.write(f"{i}\n")
            
        # Not interested in [Typedef]
        elif i.find("[Typedef]") != -1:
            break
    
    oup.close()
      

def help():
    print('\nscript_data_vocab_po.py -f <function> -i <input> -o <output>')
    print('  -f functions:')
    print('      get_names')
    print('  -i input file: in obo format')
    print('  -o output file name')
    print('\n\n')
    
#-------------------------------------------------------------------------------   
if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hf:i:o:")
    except getopt.GetoptError:
        print("\nERR: GetoptErrr")
        help() 
        sys.exit(0)

    func = infile = outfile = ""
    for opt, arg in opts:
        if opt == '-h':
            help()
            sys.exit(0)
        elif opt in ("-f"):
            func = arg              
        elif opt in ("-i"):
            infile = arg              
        elif opt in ("-o"):
            outfile = arg              
    if func == "get_names":
        get_names(infile, outfile)
    else:
        print("\nERR: Unknown function:",func)
        help()
        
        
''' 
In Windows 10:
python ..\_codes\script_data_vocab_obo.py -f get_names -i .\vocab_po\po.obo
python ..\_codes\script_data_vocab_obo.py -f get_names -i .\vocab_po\po.obo
'''
 
 