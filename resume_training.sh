#!/bin/bash

#SBATCH -p gpu22
#SBATCH -w gpu22-a100-02
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --mem=250000
#SBATCH --gres gpu:1
#SBATCH -t 0-11:59:59
#SBATCH -o out_dual/slurm_new/%j.out
#SBATCH --mail-type=time_limit
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --mail-user=ahmed.abbas@mpi-inf.mpg.de
####SBATCH --signal=SIGUSR1@90

# Make conda available:
. ~/.bashrc_private
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate LearnDBCA_new

ROOT_DIR="out_dual/MRF_PF/nobackup/v_new/v1_wo_lb_features_1_1_16_16_8_1_1_1_100_False_False_5e-3_False_2_True_True_1.0_False/"

python train_dual_ascent.py --test-precision-float --config-file ${ROOT_DIR}/config.yaml \
    MODEL.CKPT_PATH ${ROOT_DIR}/default/version_1/checkpoints/last.ckpt
exit 0