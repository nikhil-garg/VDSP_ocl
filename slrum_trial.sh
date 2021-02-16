#!/bin/bash
#SBATCH --account=def-drod1901
#SBATCH --time=0-0:3:0
#SBATCH --cpus-per-task=1
OUTDIR=~/project/out/$SLURM_JOB_ID
mkdir -p $OUTDIR
cd $SLURM_TMPDIR


module load python/3.8

virtualenv --no-download $SLURM_TMPDIR/env  # SLURM_TMPDIR is on the compute node

source $SLURM_TMPDIR/env/bin/activate

pip install matplotlib


python3 -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl

git clone https://github.com/nengo/nengo
cd nengo
pip install -e .

cd ..

git clone https://github.com/nikhil-garg/VDSP_ocl.git
cd VDSP_ocl

pip install --no-index -r requirements.txt

python mnist_multiple_exploration.py

tar xf $SLURM_TMPDIR/VDSP_ocl -C $OUTDIR
