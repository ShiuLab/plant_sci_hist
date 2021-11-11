## Get the NCBI PubMed baseline file

In `project/plant_sci_history/1_obtaining_corpus`:

```
wget -r ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/*
```

## Be familiar with Pubmed XML

In `project/plant_sci_history/1_obtaining_corpus/_test`:

```
wget https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed21n0667.xml.gz
```

Work on the test xml file in `script_test_parse_pubmed_xml.ipynb`.
