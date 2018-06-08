from agent import Agent
from monitor import interact
import gym
import numpy as np
from sklearn.model_selection import ParameterSampler

np.random.seed(42)
env = gym.make('Taxi-v2')

param_grid = {'eps': [0.007, 0.01, 0.015],
              'alpha': [0.15, 0.2, 0.25],
              'gamma': [1.0, 0.9],
              'q_update': ['sarsamax', 'exp_sarsa0']}

# Using params:  {'alpha': 0.15, 'eps': 0.01, 'gamma': 1.0}
# Episode 20000/20000 || Best average reward 8.537

param_list = list(ParameterSampler(param_grid, n_iter=24))
# rounded_list = [dict((k, round(v, 6)) for (k, v) in d.items())
#                 for d in param_list]
for params in param_list:
    print('Using params: ', params)
    agent = Agent(**params)
    avg_rewards, best_avg_reward = interact(env, agent)
