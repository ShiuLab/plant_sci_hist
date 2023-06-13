################################################################################
#
# By:   Shin-Han Shiu
# Created: 11/16/21
# For: Parsing the USDA Plants Database to get common names
#
################################################################################

import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-u', '--usda_path', 
                    help='path to USDA plants database csv',
                    required=True)
args = parser.parse_args()

cnames = {} # {common_name:[scientific name, family]}

with open(args.usda_path) as f:
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
        #print([symbol, synonym, sname, cname, fam])
        if cname != "":
            if cname not in cnames:
                cnames[cname] = [sname, fam]
            else:
                print("Redun cname:", [cname], cnames[cname], [sname,fam])        
        L = f.readline()

with open("common_names","w") as f:
    ckeys = '\n'.join(sorted(cnames.keys()))
    f.write(f"{ckeys}\n")

