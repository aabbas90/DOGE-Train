#!/bin/bash

python train_primal_rounding.py --eval-only --config-file out_primal/WORMS/lb_loss_gru_v5/v3_normx_var_correct_lo_hi_1_5_16_64_16_50_50_True_1e-3_0.0_1_1e-4_10_True/config.yaml \
    MODEL.CKPT_PATH default/version_0/checkpoints/last.ckpt