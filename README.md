# __Plant_science_history__
 For dissecting major conceptual shifts in plant science

## Quick start

For those interested in conducting a similar analysis with their favorite set of documents, see the `_example` folder with a Jupyter notebook for repeating some of the analysis with an example corpus consisted of Arabiopsis papers from 2016 to 2020.

## Repo content

This repo documents the codes used for this study and the numbers in the directory and subdirectory names reflect the chronological progression of the project.

### 1. Obtaining corpus: getting the candidate plant science records

- Getting and checking the PubMed records,
- Getting plant taxa names from the NCBI Taxonomy database and the USDA website,
- Parsing PubMed XML records.
- Testing code for miscellaneous purposes.

### 2. Text classify: classify candidate records to build plant science corpus

2.1. Preprocessing texts
2.2. Text classification with Tf and Tf-Idf-based features
2.3. Text classification with Word2Vec-based features
2.4a. Text classification with DistilBERT-based features
2.4b. Text classification with SciBERT-based features 
2.4c. Text classification with PubMedBERT-based features (which did not work)
2.5. Applying models to candidate plant science records
2.6. Analysis of prediction outcome

### 3. Key term temporal

- This directory contains first, unsuccessful attempt in looking at dynamic topic evolution based on key terms. 
- This analysis predated the use of BERTopic and did not yield results reported in the study.

### 4. Topic model: topic modeling with BERTopic

4.1. Modeling topic with BERT
4.2. Reassigning outliers closer to topics to topics
4.3. Analysis of topic model to get sizes, top words, and topical connections
4.4. Topical progression over time
4.4b. Getting records in a specific time frame for interpretation.
4.4c. Determining trend lines with time series regression, which turned out not to be particularly useful
4.4d. LOWESS fit of topic trend lines for reordering topics to more clearly visualize topical evolution
4.4e. Responding to reviewer feedback by providing the LOWESS fit as supplemental data.
4.5. Analysis of the causes for outliers
4.6. UMAP analysis to represent records in 2D

### 5. Species over time: assessing the use of species over time

5.1. Core code for analyzing species over time
5.1a. Code for finding specific taxa name in records
5.2. Analysis of genome sequenced over time
5.3. Combined analysis of species and topics over time
5.4. Code modified from 5.1 for more general use

### 6. Topic model Arabidopsis: Testing the analysis workflow on data from one species

This was adopted for the code used in the example.

### 7. Countries: combined analysis of plant science corpus and geographical information

7.1. Getting country information from PubMed records and running Nominatim
7.1a. Testing Nominatim container for HPC runs
7.1b. Saving OSM PSF files for European countries from Geofabrik
7.1c. Assessing addresses recovered without country code
7.2. Combining country info from pycountry and Nominatim
7.3. Recovering country info through brute-force search and from email addresses
7.4. Consolidating country info from all approaches
7.5. Assessing the prevalence of plant science records for each country over time

### 8. Impact: assessing impacts based on journal citation scores

8.1. Assessing impact of each topic
8.2. Assessing impact of each country

### 9. Wrap-up: additional analyses

9.1. Miscellaneous analysis to finalize the study prior to submission.
9.2. Analysis of NSF funding records 
9.3a. Reviewer suggested analysis: Cause of growth in China/India publication volume 
9.3b. Reviewer suggested analysis: Determination of false negative rate using MeSH term "Plants"
9.3c. Reviewer suggested analysis: Statistical significance of topical enrichment
9.3d. Reviewer suggested analysis: UMAP representation of plant science record evolution over time

## Requirements

The `requirement.txt` contains version information for Python packages used in the local `conda` environment the author maintained in a Windows 11 machine running Windows Subsystem for Linux. Due to the rapid evolving nature of some of the packages used and the operating system, we cannot guarantee that the version info provided will not have conflict should you choose to install all the packages used. 
