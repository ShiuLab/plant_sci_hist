## Get the NCBI PubMed baseline file

In `project/plant_sci_history/1_obtaining_corpus/pubmed`:
- Downloaded on 11/11/21. 

```
wget -r ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/*
```

## Get the NCBI taxonomy file

In `project/plant_sci_history/1_obtaining_corpus/taxonomy`:
- File date: 11/11/21

```
wget https://ftp.ncbi.nih.gov/pub/taxonomy/taxdump.tar.gz
```


## Be familiar with Pubmed XML

In `project/plant_sci_history/1_obtaining_corpus/_test`:

```
wget https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed21n0667.xml.gz
```

Work on the test xml file in `script_test_parse_pubmed_xml.ipynb`.
