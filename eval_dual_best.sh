#!/bin/bash

#SBATCH -p recon
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

NUM_DUAL_ITR_TEST=100
NUM_ROUNDS_TEST=100
DUAL_IMPROVEMENT_SLOPE_TEST=1e-9

OUTPUT_ROOT_DIR="out_dual/WORMS/nobackup/vf/v7_double_bs2_1_1_16_16_8_1_20_20_20_True_True_1e-4_False_2_True_True_False/"
OUT_REL_DIR='test_double_final_woes_'${NUM_DUAL_ITR_TEST}_${NUM_ROUNDS_TEST}_${DUAL_IMPROVEMENT_SLOPE_TEST}

echo "OUTPUT_ROOT_DIR: "${OUTPUT_ROOT_DIR}
echo "OUT_REL_DIR: "${OUT_REL_DIR}

#--test-non-learned
python train_dual_ascent.py --eval-only \
    --config-file ${OUTPUT_ROOT_DIR}/config_best.yaml \
    TEST.NUM_DUAL_ITERATIONS ${NUM_DUAL_ITR_TEST} \
    TEST.NUM_ROUNDS ${NUM_ROUNDS_TEST} \
    TEST.DUAL_IMPROVEMENT_SLOPE ${DUAL_IMPROVEMENT_SLOPE_TEST} \
    OUTPUT_ROOT_DIR ${OUTPUT_ROOT_DIR} \
    OUT_REL_DIR ${OUT_REL_DIR}

exit 0