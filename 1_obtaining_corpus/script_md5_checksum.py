################################################################################
#
# By:   Shin-Han Shiu
# Date: 11/15/21
# For: Check MD5 hash of downloaded files
#   The codes used are partly from:
#     https://www.quickprogrammingtips.com/python/how-to-calculate-md5-hash-of-a-file-in-python.html
#
################################################################################

import argparse
import hashlib
from pathlib import Path
from os import listdir

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path_to_files', 
                    help='path to baseline and md5 files')
args = parser.parse_args()

file_list = listdir(args.path_to_files)
md5 = 0
for file in file_list:
    if not file.endswith(".md5"):
        print(file,)
        md5_hash = hashlib.md5()
        filename = Path(args.path_to_files) / file
        with open(filename,"rb") as f:
            # Read and update hash in chunks of 4K
            for byte_block in iter(lambda: f.read(4096),b""):
                md5_hash.update(byte_block)
                
        md5_hex = md5_hash.hexdigest()  

        # MD5 file looks like:
        # MD5(pubmed21n1059.xml.gz)= 95a84b5ec061f995bc9b5279aa251c83
        md5_file = Path(args.path_to_files) / (file + ".md5")
        md5 = open(md5_file).readline().strip().split("= ")[-1]
        if md5 != md5_hex:
            print(" checksum failed:", md5, md5_hex)
        else:
            print()