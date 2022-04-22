#!/bin/bash

#SBATCH -p gpu20
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --mem=200000
#SBATCH --gres gpu:1
#SBATCH -t 0-05:59:59
#SBATCH -o out_dual/slurm_new/%j.out
####SBATCH --signal=SIGUSR1@90

# Make conda available:
. ~/.bashrc_private
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate LearnDBCA

export OMP_NUM_THREADS=16
OUTPUT_ROOT_DIR="out_dual/MIS/nobackup/vf/v2_free_u_1_1_16_16_8_1_20_20_5_True_1e-3_False_1_True_True/"
OUTPUT_DIR_NEW="MIS/nobackup/vf/v2_free_u_1_1_16_16_8_1_20_20_5_True_1e-3_False_1_True_True/retrain_wo_norm_1e3/"

#--test-non-learned
python train_dual_ascent.py \
    --config-file ${OUTPUT_ROOT_DIR}/config.yaml \
    OUT_REL_DIR ${OUTPUT_DIR_NEW}

exit 0