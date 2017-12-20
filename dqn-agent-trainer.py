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

def call_video(episode_id):
    return True
    if episode_id < 10:
        return True
    return np.random.random(1) <= 0.5

#env = gym.make('FlappyBird-v0')
env = gym.make('Breakout-v0')
#env = Monitor(env, './recordings/FlappyBird-v0', force=True, video_callable=call_video)
env = Monitor(env, './recordings/Breakout-v0', force=True, video_callable=call_video)
#agent = DQN(2, 5000)
agent = DQN(4, 25000)
#agent.loadWeights('flappy.h5')
agent.loadWeights('breakout.h5')
eps = 0
meps = 0
decay = 0
episode = 0
while True:
    observation = env.reset()
    obs = helper.convert_frame(observation)
    s = helper.make_pie(obs, 4)
    score = 0
    episode += 1
    while True:
        #env.render()
        action = np.random.randint(4) if np.random.random(1) <= eps else agent.makeMove(s)
        observation, reward, done, info = env.step(action)
        obs = helper.convert_frame(observation)
        s1 = helper.add_cream(s, obs)
        fragment = agent.makeFragment(s, action, reward, s1)
        agent.remember(fragment, 1)
        score += reward

        #print 'episode =', episode, 'replay memory = ', len(agent.memory)
        agent.batchTrainOnFragment(agent.sampleMiniBatch(batch_size=32), verbose=0)
        s = s1
        if done:
            print 'episode =', episode, '\nscore =', score, '\nreplay memory =', len(agent.memory), '\neps =', eps
            break
    eps = meps + ((eps - meps) * decay)
    #agent.saveWeights('flappy.h5')
    agent.saveWeights('breakout.h5')
