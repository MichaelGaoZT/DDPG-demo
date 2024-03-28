#!/usr/bin/env python
# coding=utf-8

"""
@author: Gao Ziteng
@license: MIT
@contact: e1010863@u.nus.edu
@file: main_ddpg.py
@date: 2024/3/23 下午6:02
@desc: 
"""
import gym
import numpy as np
import random
import os
import time
import torch

from agent_ddpg import DDPG_Agent

# hyperparameters
EPISODE_NUM = 100
STEP_NUM = 200
# linear decreasing epsilon
EPSILON_START = 1.0
EPSILON_END = 0.02
# total steps = 2w
EPSILON_DECAY = 10000

# init env
env = gym.make(id='Pendulum-v1')
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

agent = DDPG_Agent(STATE_DIM, ACTION_DIM) # todo

REWARD_BUFFET = np.empty(EPISODE_NUM) # store reward for drawing
# for episode
for episode_i in range(EPISODE_NUM):
    state, _ = env.reset()
    episode_reward = 0
    done = False

    #for step
    for step_i in range(STEP_NUM):
        epsilon = np.interp(x = episode_i*STEP_NUM + step_i, xp = [0, EPSILON_DECAY], fp = [EPSILON_START, EPSILON_END])
        random_sample = random.random()
        if random_sample < epsilon:
            action = np.random.uniform(low = -2, high = 2, size = ACTION_DIM)
        else:
            action = agent.get_action(state) # todo - done

        next_state, reward, done, truncation, info = env.step(action) # truncation: end episode before done (because of some limits)

        agent.replay_buffer.add_memory(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        # sample a minibatch dataset from all transitions
        # TD-Learning
        agent.update() # todo: do learning progress in agent (update A/C network) - done

        # if done
        if done:
            break

    REWARD_BUFFET[episode_i] = episode_reward
    print( f"Episode: {episode_i+1}, Reward: {round(episode_reward, 2)}")

# get current path
current_path = os.path.dirname(os.path.realpath(__file__))
model = current_path + '/models/'
timestamp =time.strftime("%Y%m%d-%H%M%S")
# save models
torch.save(agent.actor.state_dict(), model + f'ddpg_actor_{timestamp}.pth')
torch.save(agent.critic.state_dict(), model + f'ddpg_critic_{timestamp}.pth')

env.close()

# save nn_model