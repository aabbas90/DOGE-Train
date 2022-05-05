#!/bin/bash

#SBATCH -p gpu22
#SBATCH -w gpu22-a100-04
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --mem=250000
#SBATCH --gres gpu:1
#SBATCH -t 0-11:59:59
#SBATCH -o out_dual/slurm_new/%j.out
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=ahmed.abbas@mpi-inf.mpg.de
####SBATCH --signal=SIGUSR1@90

# Make conda available:
. ~/.bashrc_private
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate LearnDBCA_new

NUM_DUAL_ITR_TEST=50
NUM_ROUNDS_TEST=2000
DUAL_IMPROVEMENT_SLOPE_TEST=0.0
OUTPUT_ROOT_DIR="out_dual/QAPLIB/nobackup/v_new2/v3_mixed_prec_lstm_gpu22_bs4_1_1_16_16_8_3_5_5_500_True_True_5e-4_False_2_True_True_True_10_0.0_True/"
# choose from lipa, sko, tai, wil
TEST_BATCH='wil'
TEST_ROOT_DIR='/home/ahabbas/data/learnDBCA/cv_structure_pred/qaplib_full/test_split_large_'${TEST_BATCH}'/'
OUT_REL_DIR='debug_'${NUM_DUAL_ITR_TEST}_${NUM_ROUNDS_TEST}_${DUAL_IMPROVEMENT_SLOPE_TEST}_${TEST_BATCH}

echo "OUTPUT_ROOT_DIR: "${OUTPUT_ROOT_DIR}
echo "OUT_REL_DIR: "${OUT_REL_DIR}

# --test-primal --test-non-learned --only-test-non-learned --test-precision-float
python train_dual_ascent.py --eval-only \
    --config-file ${OUTPUT_ROOT_DIR}/config_best.yaml \
    TEST.NUM_DUAL_ITERATIONS ${NUM_DUAL_ITR_TEST} \
    TEST.NUM_ROUNDS ${NUM_ROUNDS_TEST} \
    TEST.DUAL_IMPROVEMENT_SLOPE ${DUAL_IMPROVEMENT_SLOPE_TEST} \
    OUTPUT_ROOT_DIR ${OUTPUT_ROOT_DIR} \
    OUT_REL_DIR ${OUT_REL_DIR} \
    TEST.DATA.QAP_TEST_PARAMS.root_dir ${TEST_ROOT_DIR}
exit 0