#!/bin/bash

#SBATCH -p gpu20
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --mem=200000
#SBATCH --gres gpu:1
#SBATCH -t 0-23:59:59
#SBATCH -o out_dual/slurm_new/%j.out
####SBATCH --signal=SIGUSR1@90

# Make conda available:
. ~/.bashrc_private
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate LearnDBCA

FEATURE_EXTRACTOR_DEPTH=1
DUAL_PRED_DEPTH=1
VAR_FEATURE_DIM=16
CON_FEATURE_DIM=16
EDGE_FEATURE_DIM=8
NUM_ROUNDS_WITH_GRAD=1
NUM_DUAL_ITR=1
GRAD_DUAL_ITR_MAX_ITR=0
PREDICT_OMEGA=False
NUM_ROUNDS_TRAIN=100
BASE_LR=5e-4
NUM_HIDDEN_LAYERS_EDGE=1
USE_RELATIVE_GAP_LOSS=False
USE_NET_SOLVER_COSTS=True
PREDICT_DIST_WEIGHTS=False
FREE_UPDATE=True

#--test-non-learned
python train_dual_ascent.py --config-file config_dual/config_mrf_full.py \
    OUT_REL_DIR MRF/nobackup/vf/v3_${FEATURE_EXTRACTOR_DEPTH}_${DUAL_PRED_DEPTH}_${VAR_FEATURE_DIM}_${CON_FEATURE_DIM}_${EDGE_FEATURE_DIM}_${NUM_ROUNDS_WITH_GRAD}_${NUM_DUAL_ITR}_${GRAD_DUAL_ITR_MAX_ITR}_${NUM_ROUNDS_TRAIN}_${PREDICT_OMEGA}_${PREDICT_DIST_WEIGHTS}_${BASE_LR}_${USE_RELATIVE_GAP_LOSS}_${NUM_HIDDEN_LAYERS_EDGE}_${USE_NET_SOLVER_COSTS}_${FREE_UPDATE} \
    MODEL.FEATURE_EXTRACTOR_DEPTH ${FEATURE_EXTRACTOR_DEPTH} \
    TRAIN.BASE_LR ${BASE_LR} \
    TRAIN.USE_RELATIVE_GAP_LOSS ${USE_RELATIVE_GAP_LOSS} \
    MODEL.DUAL_PRED_DEPTH ${DUAL_PRED_DEPTH} \
    MODEL.VAR_FEATURE_DIM ${VAR_FEATURE_DIM} \
    MODEL.CON_FEATURE_DIM ${CON_FEATURE_DIM} \
    MODEL.EDGE_FEATURE_DIM ${EDGE_FEATURE_DIM} \
    MODEL.NUM_HIDDEN_LAYERS_EDGE ${NUM_HIDDEN_LAYERS_EDGE} \
    TRAIN.NUM_ROUNDS_WITH_GRAD ${NUM_ROUNDS_WITH_GRAD} \
    TRAIN.NUM_DUAL_ITERATIONS ${NUM_DUAL_ITR} \
    TRAIN.GRAD_DUAL_ITR_MAX_ITR ${GRAD_DUAL_ITR_MAX_ITR} \
    MODEL.PREDICT_OMEGA ${PREDICT_OMEGA} \
    MODEL.PREDICT_DIST_WEIGHTS ${PREDICT_DIST_WEIGHTS} \
    TRAIN.NUM_ROUNDS ${NUM_ROUNDS_TRAIN} \
    MODEL.USE_NET_SOLVER_COSTS ${USE_NET_SOLVER_COSTS} \
    MODEL.FREE_UPDATE ${FREE_UPDATE}

exit 0