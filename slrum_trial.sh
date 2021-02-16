#!/bin/bash
#SBATCH --account=def-drod1901
#SBATCH --time=0-0:5:0
#SBATCH --cpus-per-task=2
OUTDIR=~/project/out/$SLURM_JOB_ID
mkdir -p $OUTDIR

cd $SLURM_TMPDIR

module load python/3.6 

virtualenv --no-download $SLURM_TMPDIR/env  # SLURM_TMPDIR is on the compute node

source $SLURM_TMPDIR/env/bin/activate


git clone https://github.com/nikhil-garg/VDSP_ocl.git
cd VDSP_ocl

pip install --no-index -r requirements.txt

python mnist_multiple_exploration.py

tar xf $SLURM_TMPDIR/VDSP_ocl -C $OUTDIR
