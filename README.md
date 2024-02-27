# Plant_science_history
 For dissecting major conceptual shifts in plant science

## Quick start

For those interested in conducting a similar analysis with their favorite set of documents, see the `_example` folder with a Jupyter notebook for repeating some of the analysis with an example corpus consisted of Arabiopsis papers from 2016 to 2020.

## Repo content

This repo documents the codes used for this study and the numbers in the directory and subdirectory names reflect the chronological progression of the project.

1. Obtaining corpus: getting the candidate plant science records
  - Getting and checking the PubMed records,
  - Getting plant taxa names from the NCBI Taxonomy database and the USDA website,
  - Parsing PubMed XML records.
  - Testing code for miscellaneous purposes.

2. Text classify: classify candidate records to build plant science corpus
  1. Preprocessing texts
  2. Text classification with Tf and Tf-Idf-based features
  3. Text classification with Word2Vec-based features
  4. Text classification with BERT-based features
    a. DistilBERT
    b. SciBERT
    c. PubMedBERT (which did not work)
  v. Applying models to candidate plant science records
  vi. Analysis of prediction outcome

3. Key term temporal
  - This directory contains first, unsuccessful attempt in looking at dynamic topic evolution based on key terms. 
  - This analysis predated the use of BERTopic and did not yield results reported in the study.

4. Topic model: topic modeling with BERTopic
  i. Modeling topic with BERT
  ii. Reassigning outliers closer to topics to topics
  iii. Analysis of topic model to get sizes, top words, and topical connections
  iv. Topical progression over time
    b. Getting records in a specific time frame for interpretation.
    c. Determining trend lines with time series regression, which turned out not to be particularly useful
    d. LOWESS fit of topic trend lines for reordering topics to more clearly visualize topical evolution
    e. Responding to reviewer feedback by providing the LOWESS fit as supplemental data.
  v. Analysis of the causes for outliers
  vi. UMAP analysis to represent records in 2D

5. Species over time: assessing the use of species over time
  i. Core code for analyzing species over time
    a. Code for finding specific taxa name in records
  ii. Analysis of genome sequenced over time
  iii. Combined analysis of species and topics over time
  iv. Code modified from 5.1 for more general use

6. Topic model Arabidopsis: Testing the analysis workflow on data from one species
  - This was adopted for the code used in the example.

7. Countries: combined analysis of plant science corpus and geographical information
  i. Getting country information from PubMed records and running Nominatim
    a. Testing Nominatim container for HPC runs
    b. Saving OSM PSF files for European countries from Geofabrik
    c. Assessing addresses recovered without country code
  ii. Combining country info from pycountry and Nominatim
  iii. Recovering country info through brute-force search and from email addresses
  iv. Consolidating country info from all approaches
  v. Assessing the prevalence of plant science records for each country over time

8. Impact: assessing impacts based on journal citation scores
  i. Assessing impact of each topic
  ii. Assessing impact of each country

9. Wrap-up: additional analyses
  i. Miscellaneous analysis to finalize the study prior to submission.
  ii. Analysis of NSF funding records 
  iii. Reviewer suggested analysis 
    a. Cause of growth in China/India publication volume 
    b. Determination of false negative rate using MeSH term "Plants"
    c. Statistical significance of topical enrichment per country
    d. UMAP representation of plant science record evolution over time

## Requirements

The `requirement.txt` contains version information for Python packages used in the local `conda` environment the author maintained in a Windows 11 machine running Windows Subsystem for Linux. Due to the rapid evolving nature of some of the packages used and the operating system, we cannot guarantee that the version info provided will not have conflict should you choose to install all the packages used. 
