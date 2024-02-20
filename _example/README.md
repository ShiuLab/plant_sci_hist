# Topic model example

## Directory content

- `README.md`: this file with instruction.
- `environment.yml`:
  - This can be used to install set up the environment.
  - Alternatively, follow the instruction within the Jupyter notebook.
- `script_topic_over_time.ipynb`:
  - The Jupyter notebook with the codes needed to run the example.
- `corpus_arabidopsis_16_20.tsv.gz`: 
  - contains PubMed records with Arabidopsis in titles or abstracts from 2016 to 2020.
  - This should be moved to `~/projects/topic_model_example` so the outputs are not generated in the repo. 

## Background knowledge needed

- Using [Jupyter notebook](https://jupyter.org/),
- Setting up [conda environment](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html#managing-python),
- Navigating file structure in the Linux/Unix environment.

## Hardware requirement

- A computer with Linux/UNIX operating system,
- A GPU that can run `pytorch 2.0.1+cu117`.

## Further analysis

The required files generated in this example can be used for further analysis. Examples include:

- UMAP clustering
  - For generating Figure 1C, the details are in `4_topic_model/script_4_6_umap.ipynb`.
- Topic network
  - For generating Figure 2A, the details are in `4_topic_model/script_4_3b_network_graph.ipynb`.
- Topic frequency over time
  - For generating Figure 3, the details are in `4_topic_model/script_4_4d_reorder_heatmap_based_on_lowess.ipynb`.
  - For ordering the topics, details are provided in the method section "Topic categories and ordering". 