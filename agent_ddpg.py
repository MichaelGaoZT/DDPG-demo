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
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque # 一个双向队列，支持各种类型，常用于存储memory

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device type is: ", device)

# hyperparameters
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64
TAU = 5e-3

# TODO
'''
including 4 classes: critic, actor, replay_buffer and ddpg_agent
ddpg_agent uses other 3 classes:
'''



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim) # Actor's output_dim = action_dim

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * 2 # tanh belongs [-1,1], *2 belongs [-2,2]
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
        x = torch.relu(self.fc2(x)) # tanh belongs [-1,1], *2 belongs [-2,2]
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add_memory(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0) #state: 1*n dimension
        self.buffer.append((state, action, reward, next_state, done)) # append seq from right

    def sample(self, batch_size):
        # *-对列表解包，zip-以tuple形式赋值
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done # state goback to n-dimention

    def __len__(self): # if buffer<mini_batch, do not return
        return len(self.buffer)

class DDPG_Agent:
    # agent contains actor+critic+buffer
    def __init__(self, state_dim, action_dim):
        # init actor
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict()) # while initiating, actor_target=actor
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        # init critic
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())# while initiating, actor_target=actor
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.replay_buffer = ReplayBuffer(MEMORY_CAPACITY)
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0) # (n,)->(1,3)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0] # numpy 转入cpu效率较高

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE: # skip
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        # to tensor
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(np.vstack(actions)).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        # update critic
        next_actions = self.actor_target(next_states)
        target_Q = self.critic_target(next_states, next_actions.detach())
        target_Q = rewards + (GAMMA * target_Q * (1 - dones))
        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        self.critic_optimizer.zero_grad() # clear last step gradients
        critic_loss.backward() # computer derivatives of the loss
        self.critic_optimizer.step() # update parameters

        # update actor (by policy-gradient method)
        actor_loss = -self.critic(states, actions).mean() # 使用mean表示 -Q期望
        self.actor_optimizer.zero_grad() # clear last step gradients
        actor_loss.backward() # computer derivatives of the loss
        self.actor_optimizer.step() # update parameters

        # update target networks of actor and critic
        for target_param, param, in zip(self.actor_target.parameters(), self.actor.parameters()): # 常用zip同步更新网络参数
            target_param.data.copy_(TAU*param.data + (1-TAU)*target_param) # param权重小，target_param权重大

        for target_param, param, in zip(self.critic_target.parameters(), self.critic.parameters()): # 常用zip同步更新网络参数
            target_param.data.copy_(TAU*param.data + (1-TAU)*target_param) # param权重小，target_param权重大