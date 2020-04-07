## Dependencies

- Python 3
- Kivy
- Pytorch
- Numpy
- Matplotlib

*I used the simple self driving car enviroment from Artificial Intelligence A-Z online course.

### Usage:

- open a new terminal
  - type
    - ```cd Machine-Learning-Experiments/DQN```
  - run 
    - ```python enviroment```


### Examples:

![Alt Text](ex1.gif)
![Alt Text](ex2.gif)

## Used DQN Algorithm

- 1- Initialize replay buffer capacity
- 2- Initialize the network with random weights
- 3- For each episode (until the enviroment is stoped):
  - 1- Initialize the starting state
  - 2- For each time step:
    - 1- Select an action using softmax
    - 2- Execute selected action in the environment
    - 3- Observe reward and the next state
    - 4- Store experience in the replay buffer
    - 5- Sample a batch from the buffer and pass it to the policy network
    - 6- Calculate the loss between output Q Values and Target Q-values
    - 7- Use L1 Loss to update the weights to minimize the loss
