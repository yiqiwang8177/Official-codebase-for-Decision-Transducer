
cd ../..
# hopper
# ablation on architecture

# full DTd
python3 experiment_transducer_exchange.py --env hopper --dataset medium-replay --bias all --learning_rate 1.5e-4 --log_to_wandb True --seed 0 --device cuda:0
# DTd-right
python3 experiment_transducer_exchange.py --env hopper --dataset medium-replay --bias b2 --learning_rate 1.5e-4 --log_to_wandb True --seed 1 --device cuda:1
# DTd-left
python3 experiment_transducer_exchange.py --env hopper --dataset medium-replay --bias b1 --learning_rate 1.5e-4 --log_to_wandb True --seed 2 --device cuda:2
# DTd-zero
python3 experiment_transducer_exchange.py --env hopper --dataset medium-replay --bias b0 --learning_rate 1.5e-4 --log_to_wandb True --seed 2 --device cuda:3

