import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

class DQNModel(nn.Module):
    
    def __init__(self, state_number, hidden_number, action_number, activation_type="relu"):
        super(DQNModel, self).__init__()
        
        # Set the size of the state and action vectors
        self.state_number = state_number
        self.action_number = action_number
        self.hidden_number = hidden_number
        
        # Non-linear functions
        self.activation_type = activation_type
        if self.activation_type == "sigmoid":
            self.sigmoid = nn.Sigmoid()
        elif self.activation_type == "tanh":
            self.tanh = nn.Tanh()
        elif self.activation_type == "relu":
            self.relu = nn.ReLU()
        
        # Fully connected layer between input and hidden layer
        self.fc1 = nn.Linear(self.state_number, 30)
        
        # Fully connected layer between hidden and output layer
        self.fc2 = nn.Linear(30, self.action_number)
        
    def forward(self, states):
        
        # Input layer to hidden layer
        out = self.fc1(states)

        # Non-linear activation function
        if self.activation_type == "sigmoid":
            out = self.sigmoid(out)
        elif self.activation_type == "tanh":
            out = self.tanh(out)
        elif self.activation_type == "relu":
            out = self.relu(out)
        
        # Hidden layer to output layer
        q_values = self.fc2(out)
        
        return q_values
    
class ReplayBuffer(object):
    
    def __init__(self, buffer_size):
        
        self.buffer_size = buffer_size
        self.buffer = []
        
    def push(self, event):
        
        # add events to the buffer
        # event = state, new state, action, reward
        self.buffer.append(event)
        
        # if buffer is full delete the first item in the buffer
        if (len(self.buffer) > self.buffer_size):
            del self.buffer[0]

    def sample(self, batch_size):
        
        # randomly sample a batch of events from the buffer
        # samples = (state1, state2, ..)(new_state1, new_state2, ..)(action1, action2, ..)(reward1, reward2, ..)
        batch = zip(*random.sample(self.buffer, batch_size))
        
        # make batch pytorch variable
        return map(lambda x: Variable(torch.cat(x, 0)), batch)

class DQN():
    
    def __init__(self, state_size, hidden_size, action_size, gamma, learning_rate, window_size, buffer_size, temperature):
       
        # set hyper-parameters
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.temperature = temperature;
        self.buffer_size = buffer_size;
        self.window_size = window_size
        
        # create a list for the reward window 
        # used to calculate the mean of the reward
        self.reward_window = []
        
        # Instantiate a DQN model 
        self.model = DQNModel(state_size, hidden_size, action_size)
        
        # Instantiate a buffer
        self.buffer = ReplayBuffer(self.buffer_size)
        
        # Adam algorithm for optimization
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        
        # initilize last state, last action and last reward values to default
        # last state = left sensor, middle sensor, right sensor, orientation, -orientation
        self.last_state = torch.Tensor(state_size).unsqueeze(0)
        
        # action can be three values 0=0, 1=20, 2=-20 Degrees
        self.last_action = 0
        
        # reward is between -1 an 1
        self.last_reward = 0
        
    def select_action(self, state):

        # use softmax to select an action
        probs_dist = nn.functional.softmax(self.model(Variable(state, volatile = True))*self.temperature)
        action = probs_dist.multinomial(num_samples=1)
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        
        # Estimated Q values
        q_values = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        
        # Get the maz of estimated next state Q values
        next_max_q = self.model(batch_next_state).detach().max(1)[0]
        
        # Calculate Target
        target_q = (self.gamma * next_max_q) + batch_reward
        
        # Loss function l1 Loss
        loss = nn.functional.smooth_l1_loss(q_values, target_q)
        
        # initilize gradients
        self.optimizer.zero_grad()
        
        # calculate the gradients
        loss.backward(retain_graph = True)
        
        # update the weights
        self.optimizer.step()
        
    def update(self, reward, state):
        
        #state = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        new_state = torch.Tensor(state).float().unsqueeze(0)
        
        # add new transition to the buffer
        self.buffer.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        
        # select an action
        action = self.select_action(new_state)
        
        sample_number = 100

        # start train the model after there are enough samples in the buffer
        if len(self.buffer.buffer) > sample_number:
            
            # sample a batch from the buffer
            batch_state, batch_next_state, batch_action, batch_reward = self.buffer.sample(sample_number)
            
            # traing the model using the batch
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward

        # update the reward window
        self.reward_window.append(reward)
        
        # if reward window is full remove the first one to keep the size fixed
        if len(self.reward_window) > self.window_size:
            del self.reward_window[0]
            
        return action
    
    # calculate the score
    def score(self):

        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    # save the trained model
    def save(self):

        torch.save({'state_dict': self.model.state_dict(), 'optimizer' : self.optimizer.state_dict(),}, 'dqn_model.pth')
    
    # load the pre trained model
    def load(self):

        checkpoint = torch.load('dqn_model.pth')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])