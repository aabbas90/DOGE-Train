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

OUTPUT_ROOT_DIR="out_dual/WORMS/v5/v5_new_configs_learned_omega_vec_1_3_16_64_16_1_20_20_False_30/"
CKPT_REL_PATH="default/version_0/checkpoints/epoch=89-step=2429.ckpt"
OUT_REL_DIR='test_on_best_val_with_time_'${NUM_DUAL_ITR_TEST}_${NUM_ROUNDS_TEST}_${DUAL_IMPROVEMENT_SLOPE_TEST}

echo "OUTPUT_ROOT_DIR: "${OUTPUT_ROOT_DIR}
echo "OUT_REL_DIR: "${OUT_REL_DIR}
echo "CKPT_REL_PATH: "${CKPT_REL_PATH}

#    --config-file config_dual/config_worms_full_avg_test.py \
#    --config-file ${OUTPUT_ROOT_DIR}/config.yaml \
python train_dual_ascent.py --eval-only --test-non-learned \
    --config-file config_dual/config_worms.py \
    TEST.NUM_DUAL_ITERATIONS ${NUM_DUAL_ITR_TEST} \
    TEST.NUM_ROUNDS ${NUM_ROUNDS_TEST} \
    TEST.DUAL_IMPROVEMENT_SLOPE ${DUAL_IMPROVEMENT_SLOPE_TEST} \
    OUTPUT_ROOT_DIR ${OUTPUT_ROOT_DIR} \
    OUT_REL_DIR ${OUT_REL_DIR} \
    MODEL.CKPT_PATH ${CKPT_REL_PATH}

exit 0