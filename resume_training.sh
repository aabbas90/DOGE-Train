#!/bin/bash

#SBATCH -p gpu22
#SBATCH -w gpu22-a100-01
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --mem=250000
#SBATCH --gres gpu:1
#SBATCH -t 0-23:59:59
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
conda activate LearnDBCA_new #_old_glib

ROOT_DIR="out_dual/MIS/nobackup/vf/v7_lstm_1_1_16_16_8_3_20_20_20_True_True_1e-3_False_2_True_True_0.0_False_False_True/"

python train_dual_ascent.py --config-file ${ROOT_DIR}/config.yaml --test-precision-float \
    MODEL.CKPT_PATH ${ROOT_DIR}/default/version_0/checkpoints/last.ckpt
exit 0