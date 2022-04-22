#!/bin/bash

#SBATCH -p gpu20
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --mem=200000
#SBATCH --gres gpu:1
#SBATCH -t 1-23:59:59
#SBATCH -o out_dual/slurm_new/%j.out
####SBATCH --signal=SIGUSR1@90

# Make conda available:
. ~/.bashrc_private
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate LearnDBCA

ROOT_DIR="out_dual/QAPLIB/nobackup/v_new/v5_learn_sub_step_es_1_1_16_16_8_1_1_0_1000_False_False_1e-3_False_2_True_False_True_5"
#ROOT_DIR="out_dual/QAPLIB/nobackup/v_new/v5_learn_sub_step_es_1_1_16_16_8_1_1_1_1000_False_False_1e-3_False_2_True_False_True_5"
#ROOT_DIR="out_dual/QAPLIB/nobackup/v_new/v5_learn_sub_step_1_1_16_16_8_1_1_0_500_False_False_1e-3_False_2_True_False_True_5/"
#ROOT_DIR="out_dual/QAPLIB/nobackup/v_new/v5_learn_sub_step_es_1_1_16_16_8_1_1_1_500_False_False_1e-3_False_2_True_False_True_5/"
#ROOT_DIR="out_dual/QAPLIB/nobackup/v_new/v5_learn_sub_step_1_1_16_16_8_1_1_0_400_False_False_1e-3_False_2_True_False_True"
#ROOT_DIR="out_dual/QAPLIB/nobackup/v_new/v5_learn_sub_step_1_1_16_16_8_1_1_0_800_False_False_1e-3_False_2_True_False_True"
python train_dual_ascent.py --config-file ${ROOT_DIR}/config.yaml MODEL.CKPT_PATH ${ROOT_DIR}/default/version_0/checkpoints/last.ckpt

exit 0