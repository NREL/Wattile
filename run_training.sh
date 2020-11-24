#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=3
##SBATCH --mem=184000
#SBATCH --time=0-00:59:00
#SBATCH --account=intelcamp21
#SBATCH --mail-user=robert.buechler@nrel.gov
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/projects/intelcamp21/repos/intelcamp20-hpc/results/Outputs/slurm-%j.out
#SBATCH --job-name=BGP_batch_test

module purge
module load conda
source activate /projects/intelcamp21/conda/oldpandas
##conda list

python -u /projects/intelcamp21/repos/intelligentcampus-pred-analytics/testing_round_GP_HPC.py > /projects/intelcamp21/repos/intelcamp20-hpc/results/Outputs/testing_round_${SLURM_JOB_ID}.txt
##python -u /projects/intelcamp21/repos/intelligentcampus-pred-analytics/entry_point.py
##python -u /projects/intelcamp21/repos/intelligentcampus-pred-analytics/get_GP_data.py > /projects/intelcamp21/repos/intelcamp20-hpc/results/Outputs/python_output_${SLURM_JOB_ID}.txt


echo "Done with sbatch script"

