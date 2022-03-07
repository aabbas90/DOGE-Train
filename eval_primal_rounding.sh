#!/bin/bash

#SBATCH -p gpu20
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --mem=200000
#SBATCH --gres gpu:1
#SBATCH -t 0-11:59:59
#SBATCH -o out_primal/slurm/%j.out
####SBATCH --signal=SIGUSR1@90

# Make conda available:
. ~/.bashrc_private
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate LearnDBCA

export OMP_NUM_THREADS=16

NUM_DUAL_ITR_TEST=100
NUM_ROUNDS_TEST=100
DUAL_IMPROVEMENT_SLOPE_TEST=1e-4

OUT_REL_DIR='WORMS/v1_var_pert_1_3_16_32_16_10_10_True_1e-3_False_5e-3/default/version_3/non_learned_pr_avg_'${NUM_DUAL_ITR_TEST}_${NUM_ROUNDS_TEST}_${DUAL_IMPROVEMENT_SLOPE_TEST}
CKPT_REL_PATH='../checkpoints/last.ckpt'

echo "OUT_REL_DIR: "${OUT_REL_DIR}
echo "CKPT_REL_PATH: "${CKPT_REL_PATH}


python train_primal_rounding.py --eval-only \
    --config-file config_primal/config_worms.py \
    TEST.NUM_ROUNDS ${NUM_ROUNDS_TEST} \
    TEST.NUM_DUAL_ITERATIONS ${NUM_DUAL_ITR_TEST} \
    TEST.DUAL_IMPROVEMENT_SLOPE ${DUAL_IMPROVEMENT_SLOPE_TEST} \
    OUT_REL_DIR ${OUT_REL_DIR} MODEL.CKPT_PATH ${CKPT_REL_PATH}


# python train_primal_rounding.py --eval-only \
#     --config-file config_primal/config_all_inst.py \
#     --full-instances \
#     TEST.NUM_DUAL_ITERATIONS ${NUM_DUAL_ITR_TEST} \
#     TEST.NUM_ROUNDS ${NUM_ROUNDS_TEST} \
#     TEST.DUAL_IMPROVEMENT_SLOPE ${DUAL_IMPROVEMENT_SLOPE_TEST} \
#     OUTPUT_ROOT_DIR ${OUTPUT_ROOT_DIR} \
#     OUT_REL_DIR ${OUT_REL_DIR} \
#     MODEL.CKPT_PATH ${CKPT_REL_PATH}
exit 0