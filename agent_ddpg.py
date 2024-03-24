#!/usr/bin/env python
# coding=utf-8

"""
@author: Gao Ziteng
@license: MIT
@contact: e1010863@u.nus.edu
@file: agent_ddpg.py
@date: 2024/3/23 下午6:05
@desc: 
"""

# TODO
'''
including 4 classes: critic, actor, replay_buffer and ddpg_agent
ddpg_agent uses other 3 classes:
'''
import torch
import torch.nn as nn
from collections import deque # 一个双向队列，支持各种类型，常用于存储memory

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim) # Actor's output_dim = action_dim

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))*2 # tanh belongs [-1,1], *2 belongs [-2,2]
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim) # Q(s,a)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1) # Critic's output_dim = 1 - Q-value

    def forward(self, x, a):
        x = torch.cat([x, a] ,1)
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))*2 # tanh belongs [-1,1], *2 belongs [-2,2]
        x = self.fc3(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add_memory(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state).float()