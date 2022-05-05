#!/bin/bash

#SBATCH -p gpu22
##SBATCH -p recon
##SBATCH -w gpu22-a40-04
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --mem=250000
#SBATCH --gres gpu:1
#SBATCH -t 0-05:59:59
#SBATCH -o out_dual/slurm_new/%j.out
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=ahmed.abbas@mpi-inf.mpg.de
####SBATCH --signal=SIGUSR1@90

# Make conda available:
. ~/.bashrc_private
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate LearnDBCA_new #_old_glib

export OMP_NUM_THREADS=16
# NUM_DUAL_ITR_TEST=100
# NUM_ROUNDS_TEST=500
# DUAL_IMPROVEMENT_SLOPE_TEST=0.0

# export GRB_LICENSE_FILE='/home/ahabbas/gurobi_gpu20_29/gurobi.lic'
#OUTPUT_ROOT_DIR="out_dual/CT/nobackup/vf/v7_mixed_prec_var_start_round_1_1_16_16_8_1_5_5_100_True_True_1e-3_False_2_True_True_0.0_False/"
#v7_mixed_prec_var_start_round_1_1_16_16_8_1_20_20_20_True_True_1e-3_False_2_True_True_0.0_False
#OUT_REL_DIR='double_prec_'${NUM_DUAL_ITR_TEST}_${NUM_ROUNDS_TEST}_${DUAL_IMPROVEMENT_SLOPE_TEST}


NUM_DUAL_ITR_TEST=50
NUM_ROUNDS_TEST=100
DUAL_IMPROVEMENT_SLOPE_TEST=1e-9
OUTPUT_ROOT_DIR="out_dual/MIS/nobackup/vf/v6_mixed_prec_1_1_16_16_8_1_20_20_20_True_True_1e-3_False_1_True_True_0.0/"
OUT_REL_DIR='double_prec_correct_slope_'${NUM_DUAL_ITR_TEST}_${NUM_ROUNDS_TEST}_${DUAL_IMPROVEMENT_SLOPE_TEST}

# NUM_DUAL_ITR_TEST=100
# NUM_ROUNDS_TEST=100
# DUAL_IMPROVEMENT_SLOPE_TEST=1e-9

# OUTPUT_ROOT_DIR="out_dual/MRF/nobackup/vf/v4_mixed_prec_wo_prev_f_1_1_16_16_8_1_1_1_100_True_Tru*"
# NUM_DUAL_ITR_TEST=5
# NUM_ROUNDS_TEST=1000
# DUAL_IMPROVEMENT_SLOPE_TEST=0.0
# OUT_REL_DIR='test_'${NUM_DUAL_ITR_TEST}_${NUM_ROUNDS_TEST}_${DUAL_IMPROVEMENT_SLOPE_TEST}

# OUTPUT_ROOT_DIR="out_dual/SPP/nobackup/v2_double_no_train_1_1_16_16_8_1_5_5_200_True_True_5e-4_False_2_False_False_True_10_0.1/"
# OUT_REL_DIR='test_'${NUM_DUAL_ITR_TEST}_${NUM_ROUNDS_TEST}_${DUAL_IMPROVEMENT_SLOPE_TEST}

echo "OUTPUT_ROOT_DIR: "${OUTPUT_ROOT_DIR}
echo "OUT_REL_DIR: "${OUT_REL_DIR}

# --test-primal --test-non-learned --only-test-non-learned --test-precision-float
python train_dual_ascent.py --eval-only --test-precision-float \
    --config-file ${OUTPUT_ROOT_DIR}/config_best.yaml \
    TEST.NUM_DUAL_ITERATIONS ${NUM_DUAL_ITR_TEST} \
    TEST.NUM_ROUNDS ${NUM_ROUNDS_TEST} \
    TEST.DUAL_IMPROVEMENT_SLOPE ${DUAL_IMPROVEMENT_SLOPE_TEST} \
    OUTPUT_ROOT_DIR ${OUTPUT_ROOT_DIR} \
    OUT_REL_DIR ${OUT_REL_DIR}

exit 0