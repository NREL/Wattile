#!/bin/bash
#SBATCH --ntasks=12
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=184000
#SBATCH --time=0-01:59:00
#SBATCH --account=intelcamp21
#SBATCH --mail-user=robert.buechler@nrel.gov
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/projects/intelcamp21/repos/intelcamp20-hpc/results/Outputs/slurm-%j.out
#SBATCH --job-name=BGP_par_test

module purge
module load conda

source /nopt/nrel/apps/anaconda/5.3/etc/profile.d/conda.sh # required to find activate function see https://github.com/conda/conda/issues/7980
conda activate /projects/intelcamp21/conda/ic-lf-deploy

python -u /projects/intelcamp21/repos/intelligentcampus-pred-analytics/testing_round_GP_HPC.py > /projects/intelcamp21/repos/intelcamp20-hpc/results/Outputs/testing_round_${SLURM_JOB_ID}.txt
##python -u /projects/intelcamp21/repos/intelligentcampus-pred-analytics/entry_point.py

echo "Done with sbatch script"
