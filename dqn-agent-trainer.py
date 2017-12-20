import os, sys
import gym
from gym.wrappers import Monitor
import gym_ple #for ple-games i.e. python-learning-environment
import matplotlib.pyplot as plt
import cv2
from DQN import DQN
import numpy as np
import helper
from collections import deque

GAME = 'Breakout-v0'
WEIGHTS = 'breakout.h5'
MEMORY_SIZE = 25000
CONTROLLER_SIZE = 4
# for exploration, meps -> min-eps
eps = 0
meps = 0
decay = 0

env = gym.make(GAME)
env = Monitor(env, './recordings/' + GAME, force=True, video_callable=helper.call_video)
agent = DQN(CONTROLLER_SIZE, MEMORY_SIZE)
agent.loadWeights(WEIGHTS)

episode = 0
total_score = 0
while True:
    observation = env.reset()
    obs = helper.convert_frame(observation)
    s = helper.make_pie(obs, 4)
    score = 0
    episode += 1
    while True:
        #env.render()
        action = np.random.randint(CONTROLLER_SIZE) if np.random.random(1) <= eps else agent.makeMove(s)
        observation, reward, done, info = env.step(action)
        obs = helper.convert_frame(observation)
        s1 = helper.add_cream(s, obs)

        # <s, a, r, s'>
        if reward > 0:
            reward *= 10
        fragment = agent.makeFragment(s, action, reward, s1)
        agent.remember(fragment, 1)
        score += reward

        #print 'episode =', episode, 'replay memory = ', len(agent.memory)
        agent.batchTrainOnFragment(agent.sampleMiniBatch(batch_size=32), verbose=0)
        s = s1
        if done:
            total_score += score
            mean_score = total_score / episode
            print 'episode =', episode, '\nscore =', score, '\ntotal score =', total_score, '\nmean score =', mean_score
            print 'replay memory =', len(agent.memory), '\neps =', eps
            break
    eps = meps + ((eps - meps) * decay)
    agent.saveWeights(WEIGHTS)
