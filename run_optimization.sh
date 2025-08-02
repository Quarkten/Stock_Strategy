#!/bin/bash

# This script is a placeholder for a more sophisticated optimization suite.
# A full implementation would require a library like Optuna or Hyperopt
# to handle the parameter search and to manage the results.

# For now, this script just runs a few experiments with different parameters.

python train_rl_agent.py --agent tqc --policy mlp --reward-params '{"loss_aversion_lambda": 2.0}' --eval-log-path data/opt/run1.csv
python train_rl_agent.py --agent tqc --policy mlp --reward-params '{"tail_gain_scale": 2.5}' --eval-log-path data/opt/run2.csv
python train_rl_agent.py --agent tqc --policy transformer --reward-params '{"gain_knee_r": 1.2}' --eval-log-path data/opt/run3.csv
