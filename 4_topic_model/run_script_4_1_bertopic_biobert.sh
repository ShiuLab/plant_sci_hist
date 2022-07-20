#!/bin/bash --login
########## SBATCH Lines for Resource Request ##########

#SBATCH --time=96:00:00
#SBATCH --nodes=1    
#SBATCH --ntasks=1     
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G 
#SBATCH --job-name=biobert

########## Command Lines for Job Running ##########

conda activate bertopic
conda info --envs

cd ~/github/plant_sci_hist/4_topic_model

~/anaconda3/envs/bertopic/bin/python script_4_1_topic_model_v2.py dmis-lab/biobert-base-cased-v1.2

scontrol show job $SLURM_JOB_ID 
js -j $SLURM_JOB_ID             
