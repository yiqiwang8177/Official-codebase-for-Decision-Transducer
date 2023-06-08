# Official-codebase-for-Decision-Transducer
This is the pytorch implementation of the UAI2023 paper  "A Trajectory is Worth Three Sentences: Multimodal Transformer for Offline Reinforcement Learning"

## Overview
This repo contains full implementation of a multimodal transformer: Decision Transducer. It was designed to improve transformers performance on offline RL, by disentangling the complicated interactions between modalities (state, action, return/goal).

![image info](./architecture.png)

## Descriptions

### Gym locomotion code (gym-transducer): 
* Train a Decision Transducer (DTd) model with: 
    * Goal $G_t$ as return: experiment_transducer.py 
    * Goal $G_t$ as state-value from IQL: experiment_transducer_goal.py 
* Train a Decision Transformer (DT) model taking:
    * Return-to-go with **[original architecture]([https://link-url-here.org](https://github.com/kzl/decision-transformer/tree/master))** : experiment_dt_small.py
    * Return-to-go with DT-large in DTd paper where the model has more heads, layers, and higher dimension.

### Gym AntMaze Navigation code (gym-transducer-goal):
* Train a Decision Transducer (DTd) with:
    * Goal $G_t$ as concat(state, goal) + waypoint prediction as auxiliary task: experiment_transducer.py
* Train a  Decision Transformer (DT) model taking:
    * Return-to-go (Caveat: in sparse reward setting, Return-to-go is binary is less useful).
