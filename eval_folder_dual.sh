# !/bin/bash

COMMAND=bash
if [[ $HOSTNAME == *"slurm"* ]]; then
  COMMAND=sbatch
fi

ROOT_FOLDER="out_dual/QAPLIB/nobackup/v_new2/"
EVAL_ROOT="out_dual/QAPLIB/nobackup/eval/"
NUM_DUAL_ITR_TEST=500
NUM_ROUNDS_TEST=500
DUAL_IMPROVEMENT_SLOPE=1e-9
NAME="fixed_clip_and_time_with_feas_check"

# ROOT_FOLDER="out_dual/MRF_PF/nobackup/v_new/"
# EVAL_ROOT="out_dual/MRF_PF/nobackup/eval/"
# NUM_DUAL_ITR_TEST=100
# NUM_ROUNDS_TEST=50
# DUAL_IMPROVEMENT_SLOPE=1e-9
# NAME="fixed_clip"

for dir in ${ROOT_FOLDER}/v1_more_es_1_1_16_16_8_1_20_0_400*/     # list directories in the form "/tmp/dirname/"
do
    dir=${dir%*/}      # remove the trailing "/"
    #echo "${dir##*/}"    # print everything after the final "/"
    CONFIG_DIR=${ROOT_FOLDER}/${dir##*/}
    echo ${CONFIG_DIR}
    EVAL_DIR=${EVAL_ROOT}/${dir##*/}
    echo ${EVAL_DIR}
    $COMMAND eval_mul_dual_ascent.sh ${CONFIG_DIR}/config_best.yaml ${NUM_DUAL_ITR_TEST} ${NUM_ROUNDS_TEST} ${DUAL_IMPROVEMENT_SLOPE} ${NAME} ${EVAL_DIR}
done