#!/bin/bash

#SBATCH -p gpu20
##SBATCH --ntasks=16
#SBATCH --nodes=1
##SBATCH --mem=200000
#SBATCH --gres gpu:1
#SBATCH -t 0-23:59:59
#SBATCH -o out_primal/slurm/%j.out
####SBATCH --signal=SIGUSR1@90

# Make conda available:
. ~/.bashrc_private
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate LearnDBCA

NUM_DUAL_ITR=20000
GRAD_DUAL_ITR_MAX_ITR=50
MIN_PERTURBATION=1e-4
MAX_NUM_ROUNDS=500
BASE_LR=1e-4
FOLDER_NAME=v1_${NUM_DUAL_ITR}_${GRAD_DUAL_ITR_MAX_ITR}_${MIN_PERTURBATION}_${MAX_NUM_ROUNDS}_${BASE_LR}

echo ${FOLDER_NAME}
python train_primal_act_max.py --config-file config_primal/config_act_max_worms_train.py \
    TEST.NUM_ROUNDS ${MAX_NUM_ROUNDS} \
    TEST.NUM_DUAL_ITERATIONS ${NUM_DUAL_ITR} \
    TRAIN.GRAD_DUAL_ITR_MAX_ITR ${GRAD_DUAL_ITR_MAX_ITR} \
    TRAIN.MIN_PERTURBATION ${MIN_PERTURBATION} \
    TRAIN.BASE_LR ${BASE_LR} \
    OUT_REL_DIR WORMS/v1_act_max_worms_crops/${FOLDER_NAME}

exit 0