#!/bin/bash
#SBATCH --account=def-drod1901
#SBATCH --time=0-0:5:0
#SBATCH --cpus-per-task=2
OUTDIR=~/project/out/$SLURM_JOB_ID
mkdir -p $OUTDIR
cd $SLURM_TMPDIR

git clone https://github.com/nengo/nengo
cd nengo
pip install -e 

cd ..

git clone https://github.com/nikhil-garg/VDSP_ocl.git
cd VDSP_ocl

pip install --no-index -r requirements.txt

python mnist_multiple_exploration.py

tar xf $SLURM_TMPDIR/VDSP_ocl -C $OUTDIR
