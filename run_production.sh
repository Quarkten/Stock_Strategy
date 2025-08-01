#!/bin/bash

# Production-level training run for the TQC agent

python train_rl_agent.py \
    --agent tqc \
    --symbol SPY \
    --timeframe 5min \
    --start 20190101 \
    --end 20231231 \
    --timesteps 1000000 \
    --walk-forward \
    --train-days 252 \
    --val-days 63 \
    --step-days 63 \
    --wandb-project rl-trader-production \
    --eval-log-path data/tqc_production_trades.csv
