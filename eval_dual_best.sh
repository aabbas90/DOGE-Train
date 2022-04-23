#!/bin/bash

#SBATCH -p gpu20
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --mem=200000
#SBATCH --gres gpu:1
#SBATCH -t 0-11:59:59
#SBATCH -o out_dual/slurm/%j_eval.out
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=ahmed.abbas@mpi-inf.mpg.de
####SBATCH --signal=SIGUSR1@90

# Make conda available:
. ~/.bashrc_private
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate DevLearnDBCA

export OMP_NUM_THREADS=16

NUM_DUAL_ITR_TEST=500
NUM_ROUNDS_TEST=500
DUAL_IMPROVEMENT_SLOPE_TEST=1e-9

OUTPUT_ROOT_DIR="out_dual/QAPLIB/nobackup/v_new2/v1_more_es_1_1_16_16_8_1_20_0_400_False_False_1e-3_False_2_True_False_True_10_1.0/"
OUT_REL_DIR='test_feas_check_cuda_'${NUM_DUAL_ITR_TEST}_${NUM_ROUNDS_TEST}_${DUAL_IMPROVEMENT_SLOPE_TEST}

echo "OUTPUT_ROOT_DIR: "${OUTPUT_ROOT_DIR}
echo "OUT_REL_DIR: "${OUT_REL_DIR}

#--test-non-learned
python train_dual_ascent.py --eval-only \
    --config-file ${OUTPUT_ROOT_DIR}/config_best_scr20.yaml \
    TEST.NUM_DUAL_ITERATIONS ${NUM_DUAL_ITR_TEST} \
    TEST.NUM_ROUNDS ${NUM_ROUNDS_TEST} \
    TEST.DUAL_IMPROVEMENT_SLOPE ${DUAL_IMPROVEMENT_SLOPE_TEST} \
    OUTPUT_ROOT_DIR ${OUTPUT_ROOT_DIR} \
    OUT_REL_DIR ${OUT_REL_DIR}

#python train_dual_ascent.py --eval-only --config-file out_dual/QAPLIB/nobackup/v_new2/v1_1_1_16_16_8_1_20_20_40_True_True_1e-3_False_2_True_False_True_10/test_best_500_100_1e-4/config.yaml

exit 0