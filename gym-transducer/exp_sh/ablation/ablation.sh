
cd ../..
# hopper

python3 experiment_transducer_exchange.py --env hopper --dataset medium-replay --bias all --learning_rate 1.5e-4 --log_to_wandb True --seed 0 --device cuda:3
python3 experiment_transducer_exchange.py --env hopper --dataset medium-replay --bias all --learning_rate 1.5e-4 --log_to_wandb True --seed 1 --device cuda:3
python3 experiment_transducer_exchange.py --env hopper --dataset medium-replay --bias all --learning_rate 1.5e-4 --log_to_wandb True --seed 2 --device cuda:3
python3 experiment_transducer_exchange.py --env hopper --dataset medium-replay --bias all --learning_rate 1.5e-4 --log_to_wandb True --seed 3 --device cuda:3

# python3 experiment_transducer_exchange.py --env halfcheetah --dataset medium-replay --bias all --log_to_wandb True --seed 0 --device cuda:0
# python3 experiment_transducer_exchange.py --env halfcheetah --dataset medium-replay --bias all --log_to_wandb True --seed 1 --device cuda:0
# python3 experiment_transducer_exchange.py --env halfcheetah --dataset medium-replay --bias all --log_to_wandb True --seed 2 --device cuda:0
# python3 experiment_transducer_exchange.py --env halfcheetah --dataset medium-replay --bias all --log_to_wandb True --seed 3 --device cuda:0

python3 experiment_transducer_exchange.py --env walker2d --dataset medium-replay --bias all --log_to_wandb True --seed 0 --device cuda:3 --batch_size 128
python3 experiment_transducer_exchange.py --env walker2d --dataset medium-replay --bias all --log_to_wandb True --seed 1 --device cuda:3 --batch_size 128
python3 experiment_transducer_exchange.py --env walker2d --dataset medium-replay --bias all --log_to_wandb True --seed 2 --device cuda:3 --batch_size 128
python3 experiment_transducer_exchange.py --env walker2d --dataset medium-replay --bias all --log_to_wandb True --seed 3 --device cuda:3 --batch_size 128