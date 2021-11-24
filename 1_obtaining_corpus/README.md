# 11/23/21

## Consolidate qualified pubmed plant science records

cat pubmed/0000/*tsv pubme
d/0200/*tsv pubmed/0400/*tsv pubmed/0600/*tsv pubmed/0800/*tsv pubmed/1000/*tsv > pubmed_qualified.tsv

1497544 records.

# 11/17/21

Since 11/11/21, I pick the project backup from mid March. Based on the ground work that was done earlier, there are some majo changes:
1. Plant names: including both NCBI taxonmy, USDA Plant Database, and a few custom words related to plants. Also,
2. Pubmed xml parsing: Different flags were used. Particularly for abstract, there are abstracts are not just one paragraph but multuple. This is dealt with.
3. Plant name matching: Now allow compound plant names to be matched, but this makes thing a lot slower.
4. Rename existing folder to be the same as the GitHub folder names.

`mv 0_pubmed/ 1_obtaining_corpus`
`mv 1_doc2vec 2_doc2vec`
`mv pubmed_baseline/ pubmed`
`rm -rf pubmed_update/`

## Plant names

A pubmed record must contain "plant", "plants", "botany", "botanical"  (and corresponding upper case form) or one of the plant taxon name from NCBI taxonomy or USDA plant checklist. 

In `project/plant_sci_history/1_obtaining_corpus/taxonomy`:
- [NCBI taxonmy file](https://ftp.ncbi.nih.gov/pub/taxonomy/taxdump.tar.gz): file date: 11/11/21
  - For parsing the NCBI taxonomy file, codes were first developed in `script_testing_for_corpus.ipynb`.
  - After parsing with `script_get_plant_taxa.py`, an output file `Viridiplantae_33090_offspring` was generated. There are 20394 non-redundant names.

In `project/plant_sci_history/1_obtaining_corpus/usda`:
- [Complete PLANTS checklist](https://plants.usda.gov/home/downloads)
  - The file is further processed with Excel to yield a list of common names with `script_get_common_names.py`.

In `project/plant_sci_history/1_obtaining_corpus/`:

`cat taxonomy/Viridiplantae_33090_offspring usda/common_names > plant_names`

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

- Get plant related pubmed entries and do matching on a few examples

`python codes/script_parse_pubmed_xml.py -x _test3/ -p plant_names`

# 3/19/21

## Unresolved questions

1) How do parts of the knowledge network influence each other?
2) What is the measure for defining connections?
3) What are the vertices in the network? Entities?
4) What should the time resolution be?

## What this project can generate:

1) Conceptual evolution in plant science
2) What the major topics are in plant science. Documents to organize: "After 
   training, a topic model can be used to extract topics from new documents 
   (documents not seen in the training corpus)." - Gensim doc

To do:
* Count the numbers of papers for each taxa over time (cannot count on rank)
  - Rank: phylum
    - Chlorophyta  (green algae)
    - Prasinodermophyta
    - Streptophyta
  - Rank: subphylum (just on Streptophyta)
    - Chlorokybophyceae
    - Klebsormidiophyceae
    - Mesostigmatophyceae
    - Streptophytina
  - Rank: class/clade (just on Streptophytina)
    - Charophyceae
    - Coleochaetophyceae
    - Embryophyta
    - Zygnemophyceae
  - Embryophyta (cldes)
    - Anthocerotophyta (hornworts)
    - Bryophyta (mosses)
    - Marchantiophyta (liverworts)
    - Tracheophyta
  - Tracheophyta
    - Euphyllophyta
    - Lycopodiopsida (clubmosses)
  - Euphyllophyta
    - Polypodiopsida
    - Spermatophyta
  - Spermatophyta
    - Acrogymnospermae
    - Magnoliopsida (flowering plants)

  - At the genus level
* Figure out what plant science terms are most prominent per decade.

Key numbers:
* Plant science abstract # from 1961 to 2020: 1,690,681

# 3/18/21

## Count number of papers over time

Realize that we need a time cutoff. Use 1961 to 2020. Modify:
script_0_get_abs.combined_qualified() to enforce the range. Also change it so it
can take gzip inputs.

python ../_scripts/script_0_get_abs.py -f combined_qualified -d qualified_round1

 entries: 1690681
 excluding: 1198 too old, 2937 too new

gzip pubmed_qualified

## Number of papers in pubmed over time

`cd /mnt/home/shius/project/plant_sci/1_doc2vec
`conda activate nlp
`python ../_scripts/script_1_doc2vec.py -f check_date_range -i pubmed_qualified.gz`

Convert dates to numbers

`python ../_scripts/script_1_doc2vec.py -f date_to_number -i pubmed_qualified.gz.date_count`

In Raibowtroll:~/projects/plant_sci/_doc_for_analysis

`scp shius@hpc.msu.edu:/mnt/home/shius/project/plant_sci/1_doc2vec/pubmed_qualified.gz.date_count.date_num ./`


# 3/17/21

## Get common words

Top 3000: https://www.ef.com/wwen/english-resources/english-vocabulary/top-3000-words/

Saved in ~/project/plant_sci/1_doc2vec/vocab/common3k

# 3/16/21

## Address issue with date

On 3/15/21, found that there are some dates with excessive amount of refs. E.g.,
10/2012 has 24k, surrounding month have 3-6k only. Concerned that the use of
DateCompleted is not good. Use PubDate instead.

mkdir pubmed_baseline/1000
mv pubmed_baseline/0800/pubmed21n10* pubmed_baseline/1000

Rerun batch_check_taxa_names

`sbatch run_batch_check_taxa_words_1.sh`
`sbatch run_batch_check_taxa_words_2.sh`
`sbatch run_batch_check_taxa_words_3.sh`
`sbatch run_batch_check_taxa_words_4.sh`
`sbatch run_batch_check_taxa_words_5.sh`
`sbatch run_batch_check_taxa_words_6.sh`

## Consolidate qualified files

`rm qualified_round1/*.qualified`
`mv pubmed_baseline/*/*.qualified qualified_round1/`
`python ../_scripts/script_0_get_abs.py -f combined_qualified -d qualified_round1`

KEY NUM: 1,693,210 qualified <NOT FINAL>

After running script_1_doc2vec.py -f check_date_range() below, found out that
some 2021 entries creeps into the record. Modify combined_qualified() to filter
them out. Then run again.

`python ../_scripts/script_0_get_abs.py -f combined_qualified -d qualified_round1`

KEY NUM: 1,691,879 qualified

`gzip qualified_round1/* &`
`gzip pubmed_qualified &`


# Create a soft link to qualified entry file for doc2vec. Check range

cd ../1_doc2vec/
ln -s ../0_pubmed/pubmed_qualified.gz

conda activate nlp
python ../_scripts/script_1_doc2vec.py -f check_date_range -i pubmed_qualified.gz

Key output: pubmed_qualified.gz.date_count

# Consolidate vocab - follow what was done on 3/13/21

Also, realize that it will be useful to see how quantitative science ideas are
infused into plant science. So also get math, stat, and cs dictionaries.

Rename script_data_vocab_dict.py as script_1_data_vocab_dict.py, get math, stat, cs
dictionaries:

In Rainbowtroll/media/shius/DOC/__JOB_LOCAL/Project/KG/_vocab
python ../_codes/script_1_data_vocab_dict.py -f get_topic_htmls -o vocab_mth -t mth
python ../_codes/script_1_data_vocab_dict.py -f get_topic_htmls -o vocab_sta -t sta
python ../_codes/script_1_data_vocab_dict.py -f get_topic_htmls -o vocab_com -t com
python ../_codes/script_1_data_vocab_dict.py -f parse_htmls -o vocab_mth
python ../_codes/script_1_data_vocab_dict.py -f parse_htmls -o vocab_sta
python ../_codes/script_1_data_vocab_dict.py -f parse_htmls -o vocab_com
python ../_codes/script_1_data_vocab_dict.py -f parse_htmls -o vocab_com

In HPC:~/project/plant_sci/1_doc2vec

python ../_scripts/script_1_doc2vec.py -f consolidate_vocab -d vocab


# 3/15/21

## Fix script_9_get_abs.batch_check_taxa_words()

Issues:
1) The .qualified output should have PMID in the first column but no.
2) PubDate fields are missing for some entires, need to check which ones.
3) Some records are updated in later releases, but the funciton is ignorant.
4) It takes too long to run in one process and the run gets killed.

Put files into directories in batches

In /mnt/home/shius/project/plant_sci/0_pubmed/pubmed_baseline
`mkdir 0000 0200 0400 0600 0800`
`rm *.qualified.gz *.md5`
`rm -rf pubmed`
`mv ../pubmed_update/pubmed21n10*.gz pubmed_baseline/`

`mv pubmed21n00* pubmed21n01* 0000`
`mv pubmed21n02* pubmed21n03* 0200`
`mv pubmed21n04* pubmed21n05* 0400`
`mv pubmed21n06* pubmed21n07* 0600`
`mv pubmed21n* 0800`

Run batch_check_taxa_words

`cd ~/project/plant_sci/0_pubmed
`sbatch run_batch_check_taxa_words_1.sh`
`sbatch run_batch_check_taxa_words_2.sh`
`sbatch run_batch_check_taxa_words_3.sh`
`sbatch run_batch_check_taxa_words_4.sh`
`sbatch run_batch_check_taxa_words_5.sh`
`sbatch run_batch_check_taxa_words_5b.sh`

Consolidate qualified files

`mkdir qualified_round1`
`mv pubmed_baseline/*/*.qualified qualified_round1/`

`python ../_scripts/script_0_get_abs.py -f combined_qualified -d qualified_round1`

420 redundant PMIDs
6099 no pub date
1688914 entries in pubmed_qualified

Compress file to save space

`gzip qualified_round1/* &`
`gzip pubmed_qualified &`

## Create a soft link to qualified entry file for doc2vec

`cd ../1_doc2vec/`
`ln -s ../0_pubmed/pubmed_qualified.gz`

## Check date range

`python ../_scripts/script_1_doc2vec.py -f check_date_range -i pubmed_qualified.gz`

Key output: pubmed_qualified.gz.date_count

Found that there are some dates with excessive amount of refs


# 3/13/21

## Install gensim

In HPC

`conda create --name nlp`
`conda activate nlp`
`conda install -c conda-forge gensim`

## Follow tutorials: 
https://rare-technologies.com/doc2vec-tutorial/
https://radimrehurek.com/gensim/auto_examples/#core-tutorials-new-users-start-here

## Create new working directory

`pwd`
`~/project/plant_sci/`
`mkdir 1_doc2vec`
`cd 1_doc2vec`

## Science volcab

This was done previously in July 2020. Info in:
`RainbowTrout: doc drive\__JOB_LOCAL\Project\KG`

Upload vocabs to HPC in:
/mnt/home/shius/project/plant_sci/1_doc2vec/vocab

Consolidate vocab files into a non-redundant one:

`python ../_scripts/script_1_doc2vec.py -f consolidate_vocab -d vocab`

## Get word freq of the qualified pubmed entires for words in the vocab

`ln -s ../0_pubmed/pubmed_plant.qualified.gz`

Not done, found an issue with the .qualified output. Fix it.


# 3/11/21

## What's needed from the pubmed records

0. <PMID Version="1">
1. <Journal> --> <PubDate> --> <Month>, <Year>
2. <Journal> --> <Title>
3. <ArticleTitle>
4. <AbstractText>

## Search for qualified words

Testing on one file

`python ../_scripts/script_0_get_abs.py -f check_taxa_words -x pubmed_baseline/pubmed21n1000.xml.gz -t viridiplantae_210308.xml.all_taxa`

Batch run of the baseline files

`python ../_scripts/script_0_get_abs.py -f batch_check_taxa_words -d ~/project/plant_sci/0_pubmed/pubmed_baseline -t viridiplantae_210308.xml.all_taxa -l log.pubmed_baseline_qualified`

Batch run of the update files

`python ../_scripts/script_0_get_abs.py -f batch_check_taxa_words -d ~/project/plant_sci/0_pubmed/pubmed_update -t viridiplantae_210308.xml.all_taxa -l log.pubmed_update_qualified`

Compile all qualified entries:

`cat pubmed_baseline/*.qualified pubmed_update/*.
`qualified > pubmed_plant.qualified`
`Gzip files to reduce space usage`
`gzip pubmed_baseline/*.qualified &`
`gzip pubmed_update/*.qualified &`
`gzip pubmed_plant.qualified`
`gzip viridiplantae_210308.xml`

## KEY OUTPUT: pubmed_plant.qualified

Each row: a qualified pubmed records
5 columns: PMID, Pub date (month-year), Journal title, Article title, abstract

# 3/9/21

## Taxa names

Originally plan to just get genus names. Realize that folks may refer to their
plants with other taxonomy level names. So create a new function that get all 
taxa levels below Viridiplantae.

`python ../_scripts/script_0_get_abs.py -f get_taxa_names -i viridiplantae_210308.xml`

## Get pubmed data

Therea are these baseline files (XMLs) from NCBI FTP site which contain all
pubmed records till Dec 2020 (is Dec included? No):
ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/

In ~/project/plant_sci/0_pubmed/pubmed_baseline/

`wget -r -np ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/`

There are also update files. I need this till pubmed21n1084.xml.gz to cover
till the end of 2020:
ftp://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles 

In ~/project/plant_sci/0_pubmed/pubmed_update

Exclude htmls and include specific files

`wget -r -np -R html -A "pubmed21n106*","pubmed21n107*","pubmed21n1081*","pubmed21n1082*","pubmed21n1083*","pubmed21n1084*"  ftp://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles`

In ~/project/plant_sci/0_pubmed, checksum for downloaded files

`python ../_scripts/script_0_get_abs.py -f check_md5 -d /mnt/home/shius/project/plant_sci/0_pubmed/pubmed_baseline/`

`python ../_scripts/script_0_get_abs.py -f check_md5 -d ~/project/plant_sci/0_pubmed/pubmed_update`

# 3/8/21

## Create project folder in HPC

`/mnt/home/shius/project/plant_sci`

`/mnt/home/shius/project/plant_sci/0_pubmed`

## Get and parse Viridiplantae XMLs from NCBI taxnomy database

`python ../_scripts/script_0_get_abs.py -f get_viridiplantae > viridiplantae_210308.xml`

- Realize that I can also use the taxdump data from NCBI: https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/
- But I'd need some time to figure this out. So stick to the script above.
- Parse XML to get taxa names

`python ../_scripts/script_0_get_abs.py -f get_genus_names -i viridiplantae_210308.xml`

Output: viridiplantae_210308.xml.genus_names



