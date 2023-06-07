#!/bin/bash

cd ./..

# original Decision Transformer (DT)
python3 experiment_dt_small.py --env halfcheetah --log_to_wandb True --dataset medium --warmup_steps 10000 --seed 0 --multimodal 3 --device cuda:0 

# A large DT
# experiment_dt.py

# a large DT where Retunr-to-go is replaced by a goal location in maze
# experiment_dt_goal.py

