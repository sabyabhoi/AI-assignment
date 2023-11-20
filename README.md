# Hyper-Parameter Optimization using AI Algorithms

## Group 4

| Name            | ID            |
| --------------- | ------------- |
| Ashwin Arun     | 2020A7PS1291H |
| Sabyasachi Bhoi | 2020B3A72147H |

## Abstract

The primary objective of our project is to optimize the hyper-parameters of a Long-Short Term Memory (LSTM) Model for forecasting the prices of a traded stock. The hyper-parameters which are optimized are: the window length and number of nodes in the first and second layer of the LSTM model. The optimization is performed using four different AI algorithms.

## Algorithms Used

- Genetic Algorithm
- Particle Swarm Optimization
- Simulated Annealing
- Grid Search

# Results

## Genetic Algorithm

## Particle Swarm Optimization

## Simulated Annealing

The simulated annealing algorithm found out the optimal hyper-parameters to be:

- window length = 3
- number of units in hidden layer 1 = 191
- number of units in hidden layer 2 = 6

![simulated annealing graph](results/sa.png)

- Overall loss: 25.3410

## Grid Search

The simulated annealing algorithm found out the optimal hyper-parameters to be:

- window length = 5
- number of units in hidden layer 1 = 128
- number of units in hidden layer 2 = 8

![grid search graph](results/grid_search.png)

- Overall loss: 18.2102
