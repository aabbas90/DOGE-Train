#!/bin/bash

#SBATCH -p gpu20
##SBATCH -p gpu22
##SBATCH -w gpu22-a100-02
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
conda activate LearnDBCA

export OMP_NUM_THREADS=16

# NUM_DUAL_ITR_TEST=500
# NUM_ROUNDS_TEST=20
# DUAL_IMPROVEMENT_SLOPE_TEST=1e-9
# OUTPUT_ROOT_DIR="out_dual/WORMS/nobackup/v_new//v1_mixed_wo_prev_omega_dw_1_1_16_16_8_1_20_20_20_True_True_1e-3_False_2_True_True_False/"
# CKPT_REL_PATH="out_dual/WORMS/nobackup/v_new//v1_mixed_wo_prev_omega_dw_1_1_16_16_8_1_20_20_20_True_True_1e-3_False_2_True_True_False/default/version_0/checkpoints/epoch=179-step=539.ckpt"
# OUT_REL_DIR='test_learned_pr1_'${NUM_DUAL_ITR_TEST}_${NUM_ROUNDS_TEST}_${DUAL_IMPROVEMENT_SLOPE_TEST}

# NUM_DUAL_ITR_TEST=500
# NUM_ROUNDS_TEST=100
# DUAL_IMPROVEMENT_SLOPE_TEST=0.0
# OUTPUT_ROOT_DIR="out_dual/MRF/nobackup/vf/v2_grad_fix_wclip_ckpt_lb_c_1_1_16_16_8_1_20_20_40_True_5e-4_False_1_True_True"

# NUM_DUAL_ITR_TEST=500
# NUM_ROUNDS_TEST=50
# DUAL_IMPROVEMENT_SLOPE_TEST=0.0
# OUTPUT_ROOT_DIR="out_dual/CT/nobackup/vf/v4_free_u_1_1_16_16_8_1_20_20_100_True_1e-3_False_1_True_False_True/"

NUM_DUAL_ITR_TEST=50
NUM_ROUNDS_TEST=200
DUAL_IMPROVEMENT_SLOPE_TEST=0.0
OUTPUT_ROOT_DIR="out_dual/MIS/nobackup/vf/v6_mixed_prec_1_1_16_16_8_1_20_20_20_True_True_1e-3_False_1_True_True_0.0/"
CKPT_REL_PATH="out_dual/MIS/nobackup/vf/v6_mixed_prec_1_1_16_16_8_1_20_20_20_True_True_1e-3_False_1_True_True_0.0/version_0/checkpoints/last.ckpt"
# 15228.000000 24439.5000  60.5%     -  224s 1.lp

# NUM_DUAL_ITR_TEST=5
# NUM_ROUNDS_TEST=50
# DUAL_IMPROVEMENT_SLOPE_TEST=1e-9

# OUTPUT_ROOT_DIR="out_dual/MRF_PF/nobackup/v_new/v3_large_clip_1_1_16_16_8_1_1_1_100_True_True_5e-3_False_2_True_True_0.0_False/"
# CKPT_REL_PATH=${OUTPUT_ROOT_DIR}"/default/version_1/checkpoints/epoch=35-step=143.ckpt"
# OUT_REL_DIR='test_best_'${NUM_DUAL_ITR_TEST}_${NUM_ROUNDS_TEST}_${DUAL_IMPROVEMENT_SLOPE_TEST}

echo "OUTPUT_ROOT_DIR: "${OUTPUT_ROOT_DIR}
echo "OUT_REL_DIR: "${OUT_REL_DIR}
echo "CKPT_REL_PATH: "${CKPT_REL_PATH}

# --test-on-train --test-non-learned --only-test-non-learned
python train_dual_ascent.py --eval-only --test-precision-float --test-primal \
    --config-file ${OUTPUT_ROOT_DIR}/config.yaml \
    TEST.NUM_DUAL_ITERATIONS ${NUM_DUAL_ITR_TEST} \
    TEST.NUM_ROUNDS ${NUM_ROUNDS_TEST} \
    TEST.DUAL_IMPROVEMENT_SLOPE ${DUAL_IMPROVEMENT_SLOPE_TEST} \
    OUTPUT_ROOT_DIR ${OUTPUT_ROOT_DIR} \
    OUT_REL_DIR ${OUT_REL_DIR}

exit 0