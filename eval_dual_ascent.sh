#!/bin/bash

#SBATCH -p gpu20
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --mem=200000
#SBATCH --gres gpu:1
#SBATCH -t 0-05:59:59
#SBATCH -o out_dual/slurm/%j.out
####SBATCH --signal=SIGUSR1@90

# Make conda available:
. ~/.bashrc_private
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate LearnDBCA

export OMP_NUM_THREADS=16

NUM_DUAL_ITR_TEST=1000
NUM_ROUNDS_TEST=10
DUAL_IMPROVEMENT_SLOPE_TEST=0.0

OUTPUT_ROOT_DIR='out_dual/SP/v2_wo_conv_full_1_1_16_32_16_1_10_10_False/default/version_0/'
#OUTPUT_ROOT_DIR='out_dual/WORMS/v2_conv_full_1_2_16_32_16_1_20_20_1e-5/default/version_0/'
#OUTPUT_ROOT_DIR='out_dual/WORMS/v2_conv_full_1_1_16_32_16_1_20_20_1e-5/default/version_0/'
CKPT_REL_PATH='checkpoints/last.ckpt'
OUT_REL_DIR='full_instances_test_avg/all_sp_wo_conv_hist_'${NUM_DUAL_ITR_TEST}_${NUM_ROUNDS_TEST}_${DUAL_IMPROVEMENT_SLOPE_TEST}

echo "OUTPUT_ROOT_DIR: "${OUTPUT_ROOT_DIR}
echo "OUT_REL_DIR: "${OUT_REL_DIR}
echo "CKPT_REL_PATH: "${CKPT_REL_PATH}

#    --config-file config_dual/config_worms_full_avg_test.py \
python train_dual_ascent.py --eval-only \
    --config-file config_dual/config_all_inst.py \
    TEST.NUM_DUAL_ITERATIONS ${NUM_DUAL_ITR_TEST} \
    TEST.NUM_ROUNDS ${NUM_ROUNDS_TEST} \
    TEST.DUAL_IMPROVEMENT_SLOPE ${DUAL_IMPROVEMENT_SLOPE_TEST} \
    OUTPUT_ROOT_DIR ${OUTPUT_ROOT_DIR} \
    OUT_REL_DIR ${OUT_REL_DIR} \
    MODEL.CKPT_PATH ${CKPT_REL_PATH}

exit 0