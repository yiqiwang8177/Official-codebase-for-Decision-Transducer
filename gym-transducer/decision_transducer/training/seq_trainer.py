import numpy as np
import torch
from datetime import datetime
import os
import pickle
import random
from decision_transducer.training.trainer import Trainer

class SequenceTrainer(Trainer):

    def train_step(self):

        states, true_actions, dones, rtg, timesteps, attention_mask, true_lens = self.get_batch(self.batch_size)
        
        action_target = torch.clone(true_actions)
        attn_to_be_modified = attention_mask.clone()
        state_preds, action_preds, reward_preds = self.model.forward(
            states, true_actions, rtg[:,:-1], timesteps, attention_mask=attn_to_be_modified, lens = true_lens
        )

        act_dim = action_preds.shape[2]
        nan_mask1 = action_preds.isnan()
        if True in nan_mask1:
            print( f" Found nan during training" )

        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        # nan_mask = action_preds.isnan()
        # action_target = action_target[~nan_mask]
        # action_preds = action_preds[~nan_mask]
        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()
        
        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()


    def train_step_goal(self):

        states, true_actions, dones, values, timesteps, attention_mask, true_lens = self.get_batch(self.batch_size)
        
        action_target = torch.clone(true_actions)
        attn_to_be_modified = attention_mask.clone()
        state_preds, action_preds, reward_preds = self.model.forward(
            states, true_actions, values, timesteps, attention_mask=attn_to_be_modified, lens = true_lens
        )

        act_dim = action_preds.shape[2]
        nan_mask1 = action_preds.isnan()
        if True in nan_mask1:
            print( f" Found nan during training" )

        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        # nan_mask = action_preds.isnan()
        # action_target = action_target[~nan_mask]
        # action_preds = action_preds[~nan_mask]
        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()
        
        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

        return loss.detach().cpu().item()


