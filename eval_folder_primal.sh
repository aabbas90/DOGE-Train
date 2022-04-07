# !/bin/bash

COMMAND=bash
if [[ $HOSTNAME == *"slurm"* ]]; then
  COMMAND=sbatch
fi

ROOT_FOLDER="out_primal/WORMS/nobackup/v2/"
NUM_DUAL_ITR_TEST=500
NUM_ROUNDS_TEST=20

for dir in ${ROOT_FOLDER}/*/     # list directories in the form "/tmp/dirname/"
do
    dir=${dir%*/}      # remove the trailing "/"
    #echo "${dir##*/}"    # print everything after the final "/"
    OUTPUT_ROOT_DIR=${ROOT_FOLDER}/${dir##*/}
    echo ${OUTPUT_ROOT_DIR}
    $COMMAND eval_mul_primal.sh ${OUTPUT_ROOT_DIR} ${NUM_DUAL_ITR_TEST} ${NUM_ROUNDS_TEST}
done