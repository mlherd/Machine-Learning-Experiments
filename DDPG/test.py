import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ddpg import DDPGagent
from utils import *
import flatlands
import math
import statistics

env = gym.make("Flatlands-v0")

agent = DDPGagent(env)


noise = OUNoise(env.action_space)
batch_size = 2048
rewards = []
avg_rewards = []
distance = []
avg_distances = []

for episode in range(100):
    print("Episode: " + str(episode))
    all_state = env.reset()
    point = all_state["dist_upcoming_points"][2]
    if (abs(point[1]) <= 0.0001):
        theta = math.atan(point[0])
    else:
        theta = math.atan(point[0] / point[1])
	
    state = np.array([point[0], point[1], theta])
    noise.reset()
    episode_reward = 0
	
    for step in range(500):
        env.render()
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        new_state, reward, done, _ = env.step(action) 
        agent.memory.push(state, action, reward, new_state, done)
        
        d = abs(math.sqrt(new_state[0] ** 2 + new_state[1] ** 2))
        distance.append(d)
		
        if len(agent.memory) > batch_size:
            agent.update(batch_size)        
        
        state = new_state
        episode_reward += reward
        #print(episode_reward)
        if done:
            sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
            break
	
    avg_distances.append(statistics.mean(distance))
    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))

#plt.plot(rewards)
#plt.plot(avg_rewards)
#plt.plot()
#plt.xlabel('Episode')
#plt.ylabel('Reward')

print(avg_distances)
plt.plot(avg_distances)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Avg_Distance')
plt.show()