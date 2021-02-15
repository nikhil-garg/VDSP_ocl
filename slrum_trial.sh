#!/bin/bash
#SBATCH --account=def-drod1901
#SBATCH --time=0-10:00:0
#SBATCH --cpus-per-task=8



pip install --no-index -r requirements.txt


nnictl create --config config.yml
nnictl experiment export --filename nni_log.csv --type csv