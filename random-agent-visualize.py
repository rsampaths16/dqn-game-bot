import os, sys
import gym
from gym.wrappers import Monitor
import gym_ple #for ple-games i.e. python-learning-environment
import matplotlib.pyplot as plt
import cv2

GAME = 'Breakout-v0'
env = gym.make(GAME)
print env.action_space

while True:
    env.reset()
    score = 0
    while True:
        env.render()
        observation, reward, done, info = env.step(env.action_space.sample())
        score += reward
        if done:
            print score
            break
