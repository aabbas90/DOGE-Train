#!/bin/bash

#SBATCH -p gpu20
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --mem=200000
#SBATCH --gres gpu:1
#SBATCH -t 0-23:59:59
#SBATCH -o out_primal/slurm/%j.out
####SBATCH --signal=SIGUSR1@90

# Make conda available:
. ~/.bashrc_private
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate LearnDBCA

FEATURE_EXTRACTOR_DEPTH=1
PRIMAL_PRED_DEPTH=1
VAR_FEATURE_DIM=8
CON_FEATURE_DIM=16
EDGE_FEATURE_DIM=8
NUM_DUAL_ITR=10
GRAD_DUAL_ITR_MAX_ITR=10
USE_LAYER_NORM=True

python train_primal_rounding.py \
    MODEL.FEATURE_EXTRACTOR_DEPTH ${FEATURE_EXTRACTOR_DEPTH} \
    MODEL.PRIMAL_PRED_DEPTH ${PRIMAL_PRED_DEPTH} \
    MODEL.VAR_FEATURE_DIM ${VAR_FEATURE_DIM} \
    MODEL.CON_FEATURE_DIM ${CON_FEATURE_DIM} \
    MODEL.EDGE_FEATURE_DIM ${EDGE_FEATURE_DIM} \
    MODEL.USE_LAYER_NORM ${USE_LAYER_NORM} \
    TRAIN.NUM_DUAL_ITERATIONS ${NUM_DUAL_ITR} \
    TRAIN.GRAD_DUAL_ITR_MAX_ITR ${GRAD_DUAL_ITR_MAX_ITR} \
    OUT_REL_DIR CT/v3_new_gt_${FEATURE_EXTRACTOR_DEPTH}_${PRIMAL_PRED_DEPTH}_${VAR_FEATURE_DIM}_${CON_FEATURE_DIM}_${EDGE_FEATURE_DIM}_${NUM_DUAL_ITR}_${GRAD_DUAL_ITR_MAX_ITR}_${USE_LAYER_NORM}

exit 0