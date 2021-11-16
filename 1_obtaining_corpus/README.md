
## Plant names

A pubmed record must contain "plant", "plants", "botany", "botanical" or one of the plant taxon name from NCBI taxonomy or USDA plant checklist. 

In `project/plant_sci_history/1_obtaining_corpus/taxonomy`:
- [NCBI taxonmy file](https://ftp.ncbi.nih.gov/pub/taxonomy/taxdump.tar.gz): file date: 11/11/21
  - For parsing the NCBI taxonomy file, codes were first developed in `script_test_parse_pubmed_xml.ipynb`.
  - After parsing with `script_get_plant_taxa.py`, an output file `Viridiplantae_33090_offspring` was generated. There are 20394 non-redundant names.

In `project/plant_sci_history/1_obtaining_corpus/usda`:
- [Complete PLANTS checklist](https://plants.usda.gov/home/downloads)
  - The file is further processed with Excel to yield a list of common names.

## NCBI PubMed baseline file

In `project/plant_sci_history/1_obtaining_corpus/pubmed`:
- [Baseline files](ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/) downloaded on 11/11/21. 
- Do md5 checksum with `script_md5_checksum.py`: all passed.
- Unzipped all files.


## Be familiar with Pubmed XML

In `project/plant_sci_history/1_obtaining_corpus/_test`:

```
wget https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed21n0667.xml.gz
```

Work on the test xml file in `script_test_parse_pubmed_xml.ipynb`.
