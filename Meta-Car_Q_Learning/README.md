![alt text](https://github.com/mlherd/Machine-Learning-Experiments/blob/master/Meta-Car_Q_Learning/metacar.png)

```python
#Husnu Melih Erdogan
#2019, July
#Q Learning Using Meta-Car Simulator
#Meta-Car Project
#https://www.metacar-project.com/
# Meta-Car Python Wrapper
# https://github.com/AI-Guru/gym-metacar
import sys
import gym
import random
try:
    import gym_metacar
except:
    import sys
    sys.path.append(".")
    sys.path.append("..")
    import gym_metacar
import time
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

# create an enviroment
env = gym.make("metacar-level1-discrete-v0")

env.enable_webrenderer()

# reset environment to a new, random state
env.reset()

# create a q table
# observation space is 5x5 matrix
# each item in it can have 3 different values 
# -1:grass (in my case 2:grass) 1:car 0:road

# Ex lidar reading
# 2,2,2,2,2
# 2,3,3,3,2
# 2,2,2,2,2
# 0,0,0,0,1
# 0,0,1,1,2
# state = 2222 + 23332 + 2222 + 1 + 112 

# Create Q table
# I reduced the action space from 5 to 3
Q = np.zeros([111110,3])

# define training hyper-parameters
episode_number = 1000
lr = 0.1
discount = 0.9
epsilon = 0.8

# lists for rewards and plots
reward_list = []
avg_reward_list = []
total_distance = []

# epsilon reduction amount per pesiode
reduction = (epsilon)/episode_number

# lidar2state
def getState(ls):
    return ls[0]*10000+ls[1]*1000+ls[2]*100+ls[3]*10+ls[4]

# stop and read initial lidar
# caculate initial state

for episode in range(episode_number):
    #initilize the parameters for new episode
    total_reward = 0
    reward = 0
    env.reset()
    terminate = False
    step = 0
    # each epsidoe can take max 1000 steps
    # it terminates if the car crashes or gets out of the road
    for step in range (0, 1000):
        observation, reward, done, _= env.step(4)
        env.render()
        lidar = observation["lidar"]
        lidar = np.asarray(lidar)
        lidar[lidar < 0] = 2
        lidar_sums = np.sum(lidar, axis=1)
        state = getState(lidar_sums)
        
        # epsilon greedy
        if episode<3:
            action = 0 
        elif np.random.random() < 1 - epsilon:
            action = np.argmax(Q[state, :]) 
        else:
            #action = env.action_space.sample()
            action = random.randint(0,2)
        
        # get the observation
        observation, reward, done, _= env.step(action)
        env.render()
        
        #calculate the reward
        if (reward == -1):
            reward = -100
            terminate = True
        elif action == 0:
            reward = 10
        elif (action != 0):
            reward = -2
        
        #get the new state
        lidar = observation["lidar"]
        lidar = np.asarray(lidar)
        lidar[lidar < 0] = 2
        lidar_sums = np.sum(lidar, axis=1)
        new_state = getState(lidar_sums)
        
        #Calculate the new Q value and update the Q table
        delta = Q[state, action] + lr*(reward + (discount * np.max(Q[new_state, :])) - Q[state, action])
        Q[state, action] = Q[state, action] + delta
        state = new_state
        
        total_reward = total_reward + reward
        
        if terminate == True:
            break
        
        step = step + 1
    
    if epsilon > 0:
        epsilon = epsilon - reduction
    
    reward_list.append(total_reward)
    avg_reward_list.append(total_reward/step)
    total_distance.append(step)
    print("Episode: " + str(episode) + " Total Reward: " + str(total_reward) + " Avg Reward: " + str(total_reward/step) + " Total_Steps: ", str(step))
print (total_reward)
env.close()
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-1-4211d4c4b408> in <module>
          7 # https://github.com/AI-Guru/gym-metacar
          8 import sys
    ----> 9 import gym
         10 import random
         11 try:
    

    ModuleNotFoundError: No module named 'gym'



```python
# save the q table
import pickle
with open("meta-car.pkl", 'wb') as f:
    pickle.dump(Q, f)

from pandas import DataFrame
df = DataFrame(Q)
export_csv = df.to_csv (r'export_dataframe.csv', index = None, header=True)

```


```python
#testing 
#without training or changing any values in the Q table
env.reset()
for i in range(0,10):
    env.reset()
    while True:
        env.render()
        time.sleep(0.1)
        lidar = observation["lidar"]
        lidar = np.asarray(lidar)
        lidar[lidar < 0] = 2
        lidar_sums = np.sum(lidar, axis=1)
        state = getState(lidar_sums)
        action = np.argmax(Q[state, :]) 
        # get the observation
        observation, reward, done, _= env.step(action)
    
        if (reward == -1):
            break
env.close()
```


```python
#show the results
import matplotlib.pyplot as plt
%matplotlib notebook
plt.subplot(3, 1, 1)
plt.plot(reward_list)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()

plt.subplot(3, 1, 2)
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg Reward")
plt.show()

plt.subplot(3, 1, 3)
plt.plot(total_distance)
plt.xlabel("Episode")
plt.ylabel("Distance")
plt.show()
```


```python

```
