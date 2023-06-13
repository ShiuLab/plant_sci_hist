
'''
For saving osm pbf files for European countries from geofabrik

Run in the geo conda environment.

2/25/23 by Shin-Han Shiu
3/6/23
- Turned out that Russian Federation is not in the Geofabrik europe folder even
  thought is it in the europe.html page. Download that on its own from:
  https://download.geofabrik.de/russia-latest.osm.pbf
'''

import requests, subprocess, hashlib
from bs4 import BeautifulSoup
from pathlib import Path

# From:
#https://stackoverflow.com/questions/66925001/checking-file-checksum-in-python
def md5(fname):
  hash_md5 = hashlib.md5()
  hash_md5.update( open(fname,'rb').read() )
  return hash_md5.hexdigest()

# Modified from:
#https://stackoverflow.com/questions/66925001/checking-file-checksum-in-python
def get_url_cs(url_geofabrik_europe, fname):
  url_cs = url_geofabrik_europe + f"{fname}.md5"
  req_cs = requests.get(url_cs, allow_redirects=True)
  md5_cs = req_cs.content.decode('utf-8').split()[0]
  return md5_cs

# directory to save osm pbf fles in
work_dir = Path('/home/shius/data_nominatim/continent_osm/europe/')

# Special subregions to exclude so they are not overlapping
exclude = ["alps", "britain-and-ireland", "dach"]

# Get europe osm pbf file list, modified from
#https://www.geeksforgeeks.org/extract-all-the-urls-from-the-webpage-using-python/
print("Get links")
url_geofabrik_europe = "https://download.geofabrik.de/europe/"
reqs = requests.get(url_geofabrik_europe)
soup = BeautifulSoup(reqs.text, 'html.parser')

# Download osm pbf files
print("Download osm pbf files")
for link in soup.find_all('a'):

  fname = link.get('href')
  country = fname.split("-latest")[0]

  if fname.endswith('-latest.osm.pbf') and country not in exclude:
    print("", fname, end="")

    # Check if downloaded, if downloaded see if checksum is right
    fpath      = work_dir / fname

    downloaded = 0
    if fpath.is_file():
      print(": downloaded", end="", flush=True)
      # Checksum
      md5_fp = md5(fpath)
      md5_cs = get_url_cs(url_geofabrik_europe, fname)
      if md5_fp == md5_cs:
        print(" md5 good")
        downloaded = 1
      else:
        print(" md5 not good", end="", flush=True)
        # remove file
        fpath.unlink()

    if not downloaded:
      print(": downloading", end="", flush=True)
      url = f"{url_geofabrik_europe}{fname}"
      subprocess.call(['wget', '-P', work_dir, url], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      # check md5
      md5_fp = md5(fpath)
      md5_cs = get_url_cs(url_geofabrik_europe, fname)
      if md5_fp == md5_cs:
        print(" md5 good")

