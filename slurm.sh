
#!/bin/bash
#SBATCH --account=def-someuser
#SBATCH --time=0-10:00:0
#SBATCH --cpus-per-task=8


OUTDIR= ~/project/out/$SLURM_JOB_ID

mkdir -p $OUTDIR

cd $SLURM_TMPDIR

module load python/3.6 

virtualenv --no-download $SLURM_TMPDIR/env  # SLURM_TMPDIR is on the compute node

source $SLURM_TMPDIR/env/bin/activate


git clone https://github.com/nikhil-garg/VDSP_ocl.git
cd VDSP_ocl
pip install --no-index -r ~/VDSP_ocl/requirements.txt


nnictl create --config config.yml
nnictl experiment export --filename nni_log.csv --type csv