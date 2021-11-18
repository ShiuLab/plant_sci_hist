
## Plant names

A pubmed record must contain "plant", "plants", "botany", "botanical" or one of the plant taxon name from NCBI taxonomy or USDA plant checklist. 

In `project/plant_sci_history/1_obtaining_corpus/taxonomy`:
- [NCBI taxonmy file](https://ftp.ncbi.nih.gov/pub/taxonomy/taxdump.tar.gz): file date: 11/11/21
  - For parsing the NCBI taxonomy file, codes were first developed in `script_testing_for_corpus.ipynb`.
  - After parsing with `script_get_plant_taxa.py`, an output file `Viridiplantae_33090_offspring` was generated. There are 20394 non-redundant names.

In `project/plant_sci_history/1_obtaining_corpus/usda`:
- [Complete PLANTS checklist](https://plants.usda.gov/home/downloads)
  - The file is further processed with Excel to yield a list of common names with `script_get_common_names.py`.

In `project/plant_sci_history/1_obtaining_corpus/`:
- `cat taxonomy/Viridiplantae_33090_offspring usda/common_names > plant_names`
- There is a total of 51385 names.

## NCBI PubMed baseline file

In `project/plant_sci_history/1_obtaining_corpus/pubmed`:
- [Baseline files](ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/) downloaded on 11/11/21. 
- Do md5 checksum with `script_md5_checksum.py`: all passed.
- Unzipped all files.

### Testing

In `project/plant_sci_history/1_obtaining_corpus/_test` and `_test2`:

- Work on the test xml file with `script_test_parse_pubmed_xml.ipynb`.
- Production version of xml parser is `script_parse_pubmed_xml.py`. Test parse one gzipped XML with five records.
  - The debug flag in parse_xml() is manually set to 1 for this.

`python codes/script_parse_pubmed_xml.py -x _test2 > log_test_debug`

- Iterate through multiple gzipped XMLs and generated an output with `.parsed_tsv` extension.

`python codes/script_parse_pubmed_xml.py -x _test2/`

- Get plant related pubmed entries


