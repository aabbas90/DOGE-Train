#!/bin/bash

#SBATCH -p gpu22
#SBATCH -w gpu22-a100-02
##SBATCH -p gpu20
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --mem=250000
#SBATCH --gres gpu:1
#SBATCH -t 0-11:59:59
#SBATCH -o out_dual/slurm_new/%j.out
#SBATCH --mail-type=time_limit
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --mail-user=ahmed.abbas@mpi-inf.mpg.de
####SBATCH --signal=SIGUSR1@90

# Make conda available:
. ~/.bashrc
eval "$(conda shell.bash hook)"
# Activate a conda environment:
#conda activate LearnDBCA_new
conda activate LearnDBCA_new #_old_glib

export OMP_NUM_THREADS=16

CONFIG_FILE=${1}
NUM_DUAL_ITR_TEST=${2}
NUM_ROUNDS_TEST=${3}
DUAL_IMPROVEMENT_SLOPE_TEST=${4}
NAME=${5}
OUTPUT_ROOT_DIR=${6}
ARGS=${7}
OUT_REL_DIR=${NAME}_${NUM_DUAL_ITR_TEST}_${NUM_ROUNDS_TEST}_${DUAL_IMPROVEMENT_SLOPE_TEST}

echo "OUTPUT_ROOT_DIR: "${OUTPUT_ROOT_DIR}
echo "OUT_REL_DIR: "${OUT_REL_DIR}

# --test-non-learned
python train_dual_ascent.py --eval-only ${ARGS} \
    --config-file ${CONFIG_FILE} \
    TEST.NUM_DUAL_ITERATIONS ${NUM_DUAL_ITR_TEST} \
    TEST.NUM_ROUNDS ${NUM_ROUNDS_TEST} \
    TEST.DUAL_IMPROVEMENT_SLOPE ${DUAL_IMPROVEMENT_SLOPE_TEST} \
    OUTPUT_ROOT_DIR ${OUTPUT_ROOT_DIR} \
    OUT_REL_DIR ${OUT_REL_DIR}

#    TEST.VAL_PERIOD 100000 \

exit 0