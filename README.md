# Total Fuel Oil Consumption (TFOC) minimization of a ship using Reinforcement Learning (RL)
Detailed information about this project can be found in my official thesis in the following URL: 

A short summary is provided below.

## Description
* A predictive Long Short Term Memory (LSTM) Neural Network was implemented for the accurate prediction of fuel consumption per ship's step. 
* The Q-learning, DQN and Rainbow agents were compared. 

## Comparison of agents
* **Q-learning:** Proved to be an excellent choice as long as the state-space is small. As it grows larger, the Q-learning agent as expected cannot keep up. 
* **Deep Q-Network (DQN):** It is claimed to be a great alternative for large state-spaces. However, in my experience it seemed to be extremely sensitive as regarding the hyper-parameters tuning. Also, as the state-space grew larger the range of hyper-parameters that achieved learning became smaller. After some point, it was almost impossible to tune the hyper-parameters correctly in order to achieve learning. The handling of the initialization of weights was not attempted. 
* **Rainbow:** Proved to be the best alternative for this project. It solved all the problems related to the DQN agent, while offering the same pros. 

## Challenges
* The greatest of all was the sensitivity of the hyper-parameters of the DQN agent. 
* Different approaches were used for the reward strategy. 

## Some Recommendations for Future Work
* More advanced weather modelling - a simple circular storm was used as proof of concept. 
* Better handling of the speed parameter. 
* More sophisticated hyper-parameters tuning methods should be implemented. 
