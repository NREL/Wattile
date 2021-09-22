#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0-02:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=intelcamp21
#SBATCH --mail-user=robert.buechler@nrel.gov
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/projects/intelcamp21/repos/intelcamp20-hpc/results/Outputs/slurm-%j.out
#SBATCH --job-name=Cafe_GPU

module purge
module load conda
source activate /projects/intelcamp21/conda/pytorch_gpu
##conda list

python -u /projects/intelcamp21/repos/intelligentcampus-pred-analytics/entry_point.py > /projects/intelcamp21/repos/intelcamp20-hpc/results/Outputs/testing_round_${SLURM_JOB_ID}.txt
tensorboard dev upload --logdir /projects/intelcamp21/repos/intelcamp20-hpc/results/RNN_MCafeWholeBuildingRealPowerTotal_T7-9-21 --one_shot

echo "Done with sbatch script"

