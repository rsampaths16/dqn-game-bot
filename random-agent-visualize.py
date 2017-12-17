import os, sys
import gym
from gym.wrappers import Monitor
import gym_ple #for ple-games i.e. python-learning-environment
import matplotlib.pyplot as plt
import cv2

env = gym.make('FlappyBird-v0')
print env.action_space
while True:
    env.reset()
    score = 0
    while True:
        env.render()
        observation, reward, done, info = env.step(env.action_space.sample())
        print env.action_space.sample()
        score += reward
        #observation2 = cv2.resize(observation, (84, 84))
        #observation2 = cv2.cvtColor(observation2, cv2.COLOR_RGB2GRAY)
        #cv2.imshow('how', observation2)
        #cv2.waitKey(1)
        if done:
            print observation.shape
            print score
            break
