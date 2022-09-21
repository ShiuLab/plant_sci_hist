################################################################################
#
# By:   Shin-Han Shiu
# Date: 11/15/21
# For: Getting the child taxa of a taxa from NCBI taxonomy. In this specific
#   case, I am getting the child taxa of Viridiplantae. Script is developed in
#   a juputer notebook: script_test_parse_xml.ipynb.
#
################################################################################

import argparse
from pathlib import Path

#
# For: Getting the tax_id of Viridiplantae and generate a dictionary.
# Parameters
#   names_dmp_file - The Path object to the names.dmp file from NCBI taxonomy.
#   target - Target taxon name.
# Return:
#   target_id - The NCBI taxon ID for the taxon.
#   names_dic - A dictionary with: {tax_id:{name_class:[names]}
#
def get_name_dict(names_dmp_path, target):
    target_id = ""
    names_dmp = open(names_dmp_path)
    L         = names_dmp.readline()
    names_dic = {}
    while L != "":
        L = L.strip().split("\t")
        tax_id = L[0]
        name   = L[2]
        name_c = L[6]
        if L[2] == target:
            print(f"{target} tax_id:",tax_id)
            target_id = tax_id

        if tax_id not in names_dic:
            names_dic[tax_id] = {name_c:[name]}
        elif name_c not in names_dic[tax_id]:
            names_dic[tax_id][name_c] = [name]
        else:
            names_dic[tax_id][name_c].append(name)
        L = names_dmp.readline()
    return target_id, names_dic

#
# For: Get the parent-child relationships from nodes.dmp file.
# Parameters: 
#   nodes_dmp_path - The Path object to the nodes.dmp file from NCBI taxonomy.
# Return: 
#   parent_child - A dictionary with {parent:[children]}
#
def get_parent_child(nodes_dmp_path):
    nodes_dmp    = open(nodes_dmp_path)
    L            = nodes_dmp.readline()
    rank_d       = {}
    parent_child = {}
    child_parent = {}
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
        if rank not in ["no rank", "species"]:
            if par_id not in parent_child:
                parent_child[par_id] = [tax_id]
            else:
                parent_child[par_id].append(tax_id)
            if tax_id not in child_parent:
                child_parent[tax_id] = par_id
            else:
                print(f"ERR: {tax_id} with >1 parents",
                      child_parent[tax_id], par_id)
            
        L = nodes_dmp.readline()
        
    return parent_child 

#
# For: Get the offspring of a parent.
# Parameters: 
#   p - The parent taxa ID to get children for.
#   paren_child - The dictionary returned from get_parent_child().
#   offspring - An initially empty list to append offspring IDs.
# Return: 
#   offspring - The populated offspring list.
#
def get_offspring(p, paren_child, offspring, debug=0):
    if debug:
        print(p)
    if p in paren_child:
        # Initialize c with an empty element for debugging purpose
        c = [""]
        c = paren_child[p]
        if debug:
            print("",c)
        
        offspring.extend(c)
        for a_c in c:
            get_offspring(a_c, paren_child, offspring)
    else:
        if debug:
            print(" NO CHILD")
    return offspring


################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='path to nodes.dmp and names.dmp')
parser.add_argument('-t', '--targetname', help='name of the target taxon')
args = parser.parse_args()

names_dmp_path = Path(args.path) / "names.dmp"
nodes_dmp_path = Path(args.path) / "nodes.dmp"

print('Get target ID and create a name dictionary')
target_id, names_dic = get_name_dict(names_dmp_path, args.targetname)

print("Get parent_child dictionary")
parent_child = get_parent_child(nodes_dmp_path)

print("Get offspring IDs")
offspring = get_offspring(target_id, parent_child, [])

print("Generate output")
out_file = Path(args.path) / f"{args.targetname}_{target_id}_offspring"
oup = open(out_file, "w")
redun = {}
for o in offspring:
    if o in names_dic:
        for nc in names_dic[o]: # for each name_class
            if nc != 'authority': 
                for name in names_dic[o][nc]:
                    if name not in redun:
                        oup.write(f"{name}\n")
                        redun[name] = 0
oup.close()

