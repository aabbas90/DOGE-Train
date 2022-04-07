#!/bin/bash

#SBATCH -p gpu20
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --mem=200000
#SBATCH --gres gpu:1
#SBATCH -t 0-11:59:59
#SBATCH -o out_dual/slurm/%j_eval.out
####SBATCH --signal=SIGUSR1@90

# Make conda available:
. ~/.bashrc_private
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate LearnDBCA

export OMP_NUM_THREADS=16

NUM_DUAL_ITR_TEST=500
NUM_ROUNDS_TEST=200
DUAL_IMPROVEMENT_SLOPE_TEST=0.0

#OUTPUT_ROOT_DIR="out_dual/WORMS/nobackup/v8_full_instances/v2/v3_new_env_wnorm_full_bs_2_womega_1_1_16_16_8_1_20_20_False_40_True_2e-3_True_1/"

OUTPUT_ROOT_DIR="out_dual/MRF/nobackup/vf/v1_1_2_16_16_8_1_20_20_40_True_1e-3_False_1_True/"
CKPT_REL_PATH="default/version_0/checkpoints/last.ckpt"
OUT_REL_DIR='test_'${NUM_DUAL_ITR_TEST}_${NUM_ROUNDS_TEST}_${DUAL_IMPROVEMENT_SLOPE_TEST}

echo "OUTPUT_ROOT_DIR: "${OUTPUT_ROOT_DIR}
echo "OUT_REL_DIR: "${OUT_REL_DIR}
echo "CKPT_REL_PATH: "${CKPT_REL_PATH}

#    --config-file config_dual/config_worms.py \
#    --config-file config_dual/config_worms_full_avg_test.py \
# --test-non-learned

python train_dual_ascent.py --eval-only --test-non-learned \
    --config-file ${OUTPUT_ROOT_DIR}/config.yaml \
    TEST.NUM_DUAL_ITERATIONS ${NUM_DUAL_ITR_TEST} \
    TEST.NUM_ROUNDS ${NUM_ROUNDS_TEST} \
    TEST.DUAL_IMPROVEMENT_SLOPE ${DUAL_IMPROVEMENT_SLOPE_TEST} \
    OUTPUT_ROOT_DIR ${OUTPUT_ROOT_DIR} \
    OUT_REL_DIR ${OUT_REL_DIR} \
    MODEL.CKPT_PATH ${CKPT_REL_PATH}

exit 0