import numpy as np
import torch
from datetime import datetime
import os
import pickle
import random
from decision_transformer.training.trainer import Trainer

class SequenceTrainer(Trainer):

    def train_step(self):
        states, true_actions, dones, xygoals, waypoints, timesteps, attention_mask = self.get_batch(self.batch_size)
        
        action_target = torch.clone(true_actions)
        attn_to_be_modified = attention_mask.clone()

        state_preds, action_preds, reward_preds, xy_pred = self.model.forward(
            states, true_actions, xygoals, timesteps, attention_mask=attn_to_be_modified
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        xy_pred = xy_pred.reshape(-1, 2)[attention_mask.reshape(-1) > 0]
        waypoints = waypoints.reshape(-1, 2)[attention_mask.reshape(-1) > 0]
        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )
        loss += self.loss_fn(
            None, xy_pred, None,
            None, waypoints, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()
            self.diagnostics['training/waypoint_error'] = torch.mean((xy_pred-waypoints)**2).detach().cpu().item()
        return loss.detach().cpu().item()


