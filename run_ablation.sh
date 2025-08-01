#!/bin/bash

# Default values
AGENT="tqc"
TIMESTEPS=1000
WANDB_PROJECT="rl-trader-ablation"
BASE_LOG_PATH="data/ablation"

# Experiments for loss_aversion_lambda
for val in 1.5 2.0 2.5; do
    echo "Running ablation for loss_aversion_lambda=$val"
    python train_rl_agent.py \
        --agent $AGENT \
        --walk-forward \
        --timesteps $TIMESTEPS \
        --wandb-project $WANDB_PROJECT \
        --eval-log-path "${BASE_LOG_PATH}/loss_aversion_${val}.csv" \
        --reward-params "{\"loss_aversion_lambda\": $val}"
done

# Experiments for tail_gain_scale
for val in 1.5 2.0 2.5; do
    echo "Running ablation for tail_gain_scale=$val"
    python train_rl_agent.py \
        --agent $AGENT \
        --walk-forward \
        --timesteps $TIMESTEPS \
        --wandb-project $WANDB_PROJECT \
        --eval-log-path "${BASE_LOG_PATH}/tail_gain_scale_${val}.csv" \
        --reward-params "{\"tail_gain_scale\": $val}"
done

# Experiments for gain_knee_r
for val in 1.2 1.5 1.8; do
    echo "Running ablation for gain_knee_r=$val"
    python train_rl_agent.py \
        --agent $AGENT \
        --walk-forward \
        --timesteps $TIMESTEPS \
        --wandb-project $WANDB_PROJECT \
        --eval-log-path "${BASE_LOG_PATH}/gain_knee_r_${val}.csv" \
        --reward-params "{\"gain_knee_r\": $val}"
done
