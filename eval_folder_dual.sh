# !/bin/bash

COMMAND=bash
if [[ $HOSTNAME == *"slurm"* ]]; then
  COMMAND=sbatch
fi

# ROOT_FOLDER="out_dual/CT/nobackup/vf/"
# EVAL_ROOT="out_dual/CT/nobackup/vf/"
# NUM_DUAL_ITR_TEST=100
# NUM_ROUNDS_TEST=500
# DUAL_IMPROVEMENT_SLOPE=0.0
# NAME="double_prec_"
# PREFIX="v7_mixed_prec_var_start_round_1_1_16_16_8_1_1_1_400_False_False*"
# #PREFIX="v7_mixed_prec_var_start_round_1_1_16_16_8_1_20_20_100_True_True_1e-3_*"
# #PREFIX="v7_mixed_prec_var_start_round_1_1_16_16_8_1_20_20_200_True_True_1e-3_False_2_True_True_0.*"
# ARGS="" #--test-non-learned

# ROOT_FOLDER="out_dual/WORMS/nobackup/v_new/"
# EVAL_ROOT="out_dual/WORMS/nobackup/v_new/"
# NUM_DUAL_ITR_TEST=50
# NUM_ROUNDS_TEST=200
# DUAL_IMPROVEMENT_SLOPE=1e-9
# NAME="eval_long_hist"
# PREFIX="v1_mixed_wo_prev_omega_dw_1_1_16_16_8_1_20_20_20_True_True_1e-3_False_2_True_True_Fals*"
# ARGS="--test-precision-float"

ROOT_FOLDER="out_dual/QAPLIB/nobackup/v_new2/"
EVAL_ROOT="out_dual/QAPLIB/nobackup/v_new2/"
NUM_DUAL_ITR_TEST=50
NUM_ROUNDS_TEST=1000
DUAL_IMPROVEMENT_SLOPE=0.0
NAME="double_prec_last"
#PREFIX="v3_mixed_prec_start_grad_var_100_1_1_16_16_8_1_5_5_500_"
#PREFIX="v3_mixed_prec_only_free*"
PREFIX="v3_mixed_prec_lstm_gpu22_bs4_1_1_16_16_8_3_5_5_500_True_True_5e-4_False_2_True_True_True_10_0.0_T*"
ARGS="--test-non-learned"

# ROOT_FOLDER="out_dual/MRF_PF/nobackup/v_new/"
# EVAL_ROOT="out_dual/MRF_PF/nobackup/v_new/"
# NUM_DUAL_ITR_TEST=10
# NUM_ROUNDS_TEST=200
# DUAL_IMPROVEMENT_SLOPE=1e-9
# NAME="eval"
# PREFIX="v3_1_1_16_16_8_1_1_1_100_True_True_5e-3_False_2_True_True_0.1_Fals*"
# ARGS="--test-precision-float"

# ROOT_FOLDER="out_dual/SM/nobackup/vf/"
# EVAL_ROOT="out_dual/SM/nobackup/vf/"
# NUM_DUAL_ITR_TEST=25
# NUM_ROUNDS_TEST=200
# DUAL_IMPROVEMENT_SLOPE=1e-9
# NAME="double_precision_no_es"
# PREFIX="v4_mixed_prec_1_1_16_16_8_1_5_5_50_F"
# ARGS="--test-precision-float"

# ROOT_FOLDER="out_dual/SM/nobackup/vf/"
# EVAL_ROOT="out_dual/SM/nobackup/vf/"
# NUM_DUAL_ITR_TEST=50
# NUM_ROUNDS_TEST=100
# DUAL_IMPROVEMENT_SLOPE=1e-9
# NAME="flt_no_es"

for dir in ${ROOT_FOLDER}/${PREFIX}*/     # list directories in the form "/tmp/dirname/"
do
    dir=${dir%*/}      # remove the trailing "/"
    #echo "${dir##*/}"    # print everything after the final "/"
    CONFIG_DIR=${ROOT_FOLDER}/${dir##*/}
    echo ${CONFIG_DIR}
    EVAL_DIR=${EVAL_ROOT}/${dir##*/}
    echo ${EVAL_DIR}
    $COMMAND eval_mul_dual_ascent.sh ${CONFIG_DIR}/config_best.yaml ${NUM_DUAL_ITR_TEST} ${NUM_ROUNDS_TEST} ${DUAL_IMPROVEMENT_SLOPE} ${NAME} ${EVAL_DIR} ${ARGS}
done