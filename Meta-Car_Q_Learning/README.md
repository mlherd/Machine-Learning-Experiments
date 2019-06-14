![alt text](https://github.com/mlherd/Machine-Learning-Experiments/blob/master/Meta-Car_Q_Learning/metacar.png)

```python
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

Q = np.zeros([111110,3])

# define training hyper-parameters
episode_number = 1000
lr = 0.1
discount = 0.9
epsilon = 0.8

# lists for rewards
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
        
        #Calculate Q the new value and update the Q table
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
