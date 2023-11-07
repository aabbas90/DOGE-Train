#!/bin/bash

# # Without LSTM (DOGE):
# NUM_ROUNDS_WITH_GRAD=1
# USE_LSTM_VAR=False

# # With LSTM (DOGE-M):
NUM_ROUNDS_WITH_GRAD=3
USE_LSTM_VAR=True

python train_doge.py --config-file configs/config_mis.py \
    --test-precision-float \
    --test-non-learned \
    OUT_REL_DIR MIS_new_data_${USE_LSTM_VAR}_${NUM_ROUNDS_WITH_GRAD}/ \
    MODEL.USE_LSTM_VAR ${USE_LSTM_VAR} \
    TRAIN.NUM_ROUNDS_WITH_GRAD ${NUM_ROUNDS_WITH_GRAD}

exit 0