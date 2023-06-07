#!/bin/bash

cd ./..

python3 experiment_dt_small.py --env walker2d --log_to_wandb True --dataset medium-expert --warmup_steps 10000 --seed 0 --multimodal 3 --device cuda:3 & 
python3 experiment_dt_small.py --env walker2d --log_to_wandb True --dataset medium-expert --warmup_steps 10000 --seed 1 --multimodal 3 --device cuda:3 &
python3 experiment_dt_small.py --env walker2d --log_to_wandb True --dataset medium-expert --warmup_steps 10000 --seed 2 --multimodal 3 --device cuda:3 & 
python3 experiment_dt_small.py --env walker2d --log_to_wandb True --dataset medium-expert --warmup_steps 10000 --seed 3 --multimodal 3 --device cuda:3 

python3 experiment_dt_small.py --env walker2d --log_to_wandb True --dataset medium --warmup_steps 10000 --seed 0 --multimodal 3 --device cuda:3 & 
python3 experiment_dt_small.py --env walker2d --log_to_wandb True --dataset medium --warmup_steps 10000 --seed 1 --multimodal 3 --device cuda:3 &
python3 experiment_dt_small.py --env walker2d --log_to_wandb True --dataset medium --warmup_steps 10000 --seed 2 --multimodal 3 --device cuda:3 & 
python3 experiment_dt_small.py --env walker2d --log_to_wandb True --dataset medium --warmup_steps 10000 --seed 3 --multimodal 3 --device cuda:3 &
  
  