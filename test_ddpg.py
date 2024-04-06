#!/usr/bin/env python
# coding=utf-8

"""
@author: Gao Ziteng
@license: MIT
@contact: e1010863@u.nus.edu
@file: test_ddpg.py
@date: 2024/3/23 下午6:04
@desc: 
"""
from agent_ddpg import Actor
import torch
import gym
import pygame
import os
import numpy as np


# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device type is: ", device)

# init env
env = gym.make(id='Pendulum-v1', render_mode = 'rgb_array')
STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

# load actor parameters (only actor, no need critic)# get current path
current_path = os.path.dirname(os.path.realpath(__file__))
model_path = current_path + '/models/'
actor_path = model_path + "ddpg_actor_20240406-192900.pth"

actor = Actor(STATE_DIM, ACTION_DIM).to(device)

actor.load_state_dict(torch.load(actor_path))

# visualization
def process_frame(frame):
    frame = np.transpose(frame, (1, 0 ,2))
    frame = pygame.surfarray.make_surface(frame)
    return pygame.transform.smoothscale(frame, (width, height))

pygame.init()
width, height = 600, 600
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# test main()
NUM_EPISODE = 20
NUM_STEP = 200
# for episode
for episode_1 in range(NUM_EPISODE):
    state, _ = env.reset()
    # evaluations:
    episode_reward = 0
    # for step
    for step_1 in range(NUM_STEP):
        action = actor(torch.FloatTensor(state).unsqueeze(0).to(device)).detach().cpu().numpy()[0]
        next_state, reward, done, truncation, info = env.step(action)
        state = next_state
        episode_reward += reward

        frame = env.render()
        frame = process_frame(frame)
        screen.blit(frame, (0, 0))
        pygame.display.flip()
        clock.tick(60) # 60Hz

    print(f'Episode {episode_1}, Reward {episode_reward}')