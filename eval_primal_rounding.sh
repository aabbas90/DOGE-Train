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

NUM_DUAL_ITR_TEST=500
NUM_ROUNDS_TEST=20
DUAL_IMPROVEMENT_SLOPE_TEST=1e-6

OUTPUT_ROOT_DIR="out_primal/WORMS/nobackup/v2/v1_3_1_16_32_8_1_500_20_True_1e-3_0.0_5_1e-4_5_True_0.1/"
CKPT_REL_PATH="default/version_0/checkpoints/last.ckpt"
OUT_REL_DIR='test_'${NUM_DUAL_ITR_TEST}_${NUM_ROUNDS_TEST}_${DUAL_IMPROVEMENT_SLOPE_TEST}

echo "OUTPUT_ROOT_DIR: "${OUTPUT_ROOT_DIR}
echo "OUT_REL_DIR: "${OUT_REL_DIR}
echo "CKPT_REL_PATH: "${CKPT_REL_PATH}

#--test-non-learned
python train_primal_rounding.py --eval-only \
    --config-file ${OUTPUT_ROOT_DIR}/config.yaml \
    TEST.NUM_DUAL_ITERATIONS ${NUM_DUAL_ITR_TEST} \
    TEST.NUM_ROUNDS ${NUM_ROUNDS_TEST} \
    TEST.DUAL_IMPROVEMENT_SLOPE ${DUAL_IMPROVEMENT_SLOPE_TEST} \
    OUTPUT_ROOT_DIR ${OUTPUT_ROOT_DIR} \
    OUT_REL_DIR ${OUT_REL_DIR} \
    MODEL.CKPT_PATH ${CKPT_REL_PATH}

exit 0