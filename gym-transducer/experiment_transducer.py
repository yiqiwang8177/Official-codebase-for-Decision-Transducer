import gym
import numpy as np
import pandas as pd # import it before torch to avoid bugs
import torch
import transformers
import wandb

import argparse
import pickle
import random
import sys

from decision_transducer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transducer.models.decision_transducer import DecisionTransducer
from decision_transducer.models.mlp_bc import MLPBCModel

from decision_transducer.training.act_trainer import ActTrainer
from decision_transducer.training.seq_trainer import SequenceTrainer

from utils import set_seed, set_seed2

import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 10**6

def trajecotry_stats(trajs):
    """traj is a list, each element is a dict"""
    # import pandas as pd
    lengths = []
    for traj in trajs:
        length = len(traj['rewards'])
        lengths.append(length)
    print(pd.Series(lengths).describe())

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
):

    device = variant.get('device', 'cuda')
    print('Device is:', device)
    num_eval_episodes = variant['num_eval_episodes']

    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    bias_mode = variant['bias']
    norm_mode = variant['norm_mode']
    c_mode = variant['comb']
    
    group_name = f'{exp_prefix}-{env_name}-{dataset}' + f'-bias-{bias_mode}-{norm_mode}'
    if bias_mode != 'b0':
        group_name += f'-{c_mode}'
    train_seed = variant['seed']
    exp_prefix = f'{group_name}-seed-{train_seed}-{random.randint(int(1e5), int(1e6) - 1)}'

    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        # env_targets = [3600,]  
        # env_targets = [3000, 2400, 1200, 600]
        env_targets = [6000, 4500, 2400, 3000, 2300, 1800, 1200, 600]
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        # env_targets = [6000,]#  6000]
        # env_targets = [6000, 4000]
        env_targets = [10000, 8000, 4000, 2000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        # env_targets = [5000] #, 2500]
        env_targets = [9000, 7000, 4000, 3000, 2000, 1000]
        scale = 1000.
    else:
        raise NotImplementedError

    if model_type == 'bc' or variant['bias'] == 'b0':
        env_targets = env_targets[:1]  # since BC ignores target, no need for different evaluations

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # load dataset
    dataset_path = f'./../../data/{env_name}-{dataset}-v2.pkl'
    
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)
        trajecotry_stats(trajectories )
        # trajecotries is a list.
        # ach element is a trajecotry dict with keys: observations, next observations, rewards terminals
    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]] # number of steps of the trajectory with highest return
    ind = len(trajectories) - 2 # -2 b.c. last ind is len-1 and we also skip the biggest traj (since we alread include it in timesteps)
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # reweight sampling by using timesteps of each trajectory instead of uniform
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])
    
    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps of each trajecotry
        )

        s, a, d, rtg, timesteps, mask, lens =  [], [], [], [], [], [], []

        for i in range(batch_size):
            # process each trajectory (batch_inds[i]) of a batch
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            # si is the starting timestep
            si = random.randint(0, traj['rewards'].shape[0] - 1)
            true_len = len( traj['observations'][si:si + max_len] )
            lens.append(true_len)
            # an extra dimension is added at dimension 0, the rest are the same
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))

            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            # return to go
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization.
            # left pad if Gpt2, right pad for pytorch transformer
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([ s[-1], np.zeros((1, max_len - tlen, state_dim))], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([ a[-1], np.ones((1, max_len - tlen, act_dim)) * -10.], axis=1)
            d[-1] = np.concatenate([ d[-1], np.ones((1, max_len - tlen)) * 2], axis=1)
            rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, max_len - tlen, 1)), ], axis=1) / scale

            timesteps[-1] = np.concatenate([ timesteps[-1], np.zeros((1, max_len - tlen))], axis=1)
            # 1s are what we want to attend. 0s denotes masking due to padding
            mask.append(np.concatenate([ np.ones((1, tlen)), np.zeros((1, max_len - tlen)) ], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, d, rtg, timesteps, mask, lens

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                            seed = _, # using 0,1,2 as seeds
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
                print("All returns:", returns)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    if model_type == 'dt':
        model = DecisionTransducer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            pdrop=variant['dropout'],
            bias_mode = bias_mode,
            norm_mode = norm_mode,
            c_mode = c_mode,
            modality_emb = variant['modality_emb'],
            norm_joint = variant['norm_joint'],
            join_all = variant['join_all']
        )
        print(f"# of params: {count_parameters(model)} M")
    else:
        raise NotImplementedError

    if variant['load_model'] != "NO":
        load_path = os.path.join(".", "..", "..", "saved_model", variant['load_model'] )
        model.load_state_dict(torch.load(load_path, map_location=torch.device(device) ))
        print("Model Load Sucess!")

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    if dataset == "None":#'medium-expert':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps+1)/warmup_steps, 1) 
        )
    else:
        lr_ratio = 0.5 
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps+1)/warmup_steps, 1) if steps < warmup_steps else lr_ratio + (1-lr_ratio) * ( 0.9997 ** (steps-warmup_steps) )
        )
    # scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps = variant['num_steps_per_iter'] * variant['max_iters'])

    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch, # function to process each trajectory from a batch
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project= f'DTd-sensitive-left-{env_name}-{dataset}',
            config=variant
        )
        wandb.watch(model)
    # tracking the best model of each target Rtg so far
    best_rtgs = [0,0]
    # model_target0 = f"best_model_of_{env_targets[0]}"
    # model_target1 = f"best_model_of_{env_targets[1]}"
    # model_targets = [model_target0, model_target1]
    for iter in range(variant['max_iters']):
        print("\tIteration:", iter)
        plot = {}
        # logs, plot = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, plot_dict = plot, print_logs=True)
        logs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, plot_dict = plot, print_logs=True)
        if log_to_wandb:
            wandb.log(logs)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transducer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default= 10000)
    parser.add_argument('--num_eval_episodes', type=int, default=3)
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--num_steps_per_iter', type=int, default=250)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--load_model', type=str, default='NO')
    parser.add_argument('--save_model', type=str, default='NO')

    parser.add_argument('--bias', type=str, default="all")
    parser.add_argument('--comb', type=str, default="c22")
    parser.add_argument('--norm_mode', type=str, default="n3")
    parser.add_argument('--modality_emb', type=int, default= 3)
    # if need it False, just don't specify --norm_joint
    parser.add_argument('--norm_joint', action ='store_true')
    parser.add_argument('--join_all', action ='store_true')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--seed', type=int, default= 1)

    args = parser.parse_args()

    batch = vars(args)['batch_size']
    modality_emb = vars(args)['modality_emb']
    dropout = vars(args)['dropout']
    norm_joint = vars(args)['norm_joint']
    norm_prefix = 'has' if norm_joint else 'no'
    join_all = 'all' if vars(args)['join_all'] else 'sa' 

    seed = vars(args)['seed']
    lr = vars(args)['learning_rate']
    set_seed(seed) 
    # set_seed2(seed)

    # experiment(f'DTD-{batch}-10k-warm-decay-{lr}-lr-prenorm-3-layer-encoders-{ norm_prefix }ln-postnorm-join-2head-emb-{modality_emb}', variant=vars(args))
    experiment(f'DTD-{batch}-10k-warm-decay-{lr}-lr', variant=vars(args))
    # ablation usage
    # no lr decay for med-expert
    # experiment(f'Ablation-DTD-{batch}-10kw-{lr}-lr-decay-prenorm3layerencdrs-{ norm_prefix }ln-postnorm', variant=vars(args))

    # 
    # experiment(f'ReRun-DTD-{batch}-10kwarm-{lr}-lr-decay', variant=vars(args))
    # experiment(f'NewSeedReRun-DTD-{batch}-10kwarm-{lr}-lr-decay', variant=vars(args))