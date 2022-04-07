#!/bin/bash

#SBATCH -p gpu20
##SBATCH --ntasks=16
#SBATCH --nodes=1
##SBATCH --mem=200000
#SBATCH --gres gpu:1
#SBATCH -t 1-23:59:59
#SBATCH -o out_dual/slurm/%j_worms_act_max.out
####SBATCH --signal=SIGUSR1@90

# Make conda available:
. ~/.bashrc_private
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate LearnDBCA

NUM_DUAL_ITR=20
GRAD_DUAL_ITR_MAX_ITR=20
MAX_NUM_ROUNDS=500
BASE_LR=1.0
FOLDER_NAME=v3_full_grad_${NUM_DUAL_ITR}_${GRAD_DUAL_ITR_MAX_ITR}_${MAX_NUM_ROUNDS}_${BASE_LR}

echo ${FOLDER_NAME}
#
python train_dual_act_max.py --config-file config_dual/config_act_max.py \
    TEST.NUM_ROUNDS ${MAX_NUM_ROUNDS} \
    TEST.NUM_DUAL_ITERATIONS ${NUM_DUAL_ITR} \
    TRAIN.GRAD_DUAL_ITR_MAX_ITR ${GRAD_DUAL_ITR_MAX_ITR} \
    TRAIN.BASE_LR ${BASE_LR} \
    OUT_REL_DIR WORMS/nobackup/v1_act_max/${FOLDER_NAME}

exit 0