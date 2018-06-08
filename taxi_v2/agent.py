import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self, nA=6, eps=0.005, alpha=0.01, gamma=1.0, q_update='sarsamax'):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.q_update = q_update

    def greedy_policy(self, state):
        max_a_idx = np.argmax(self.Q[state])
        policy = np.ones(self.nA) * self.eps / self.nA
        policy[max_a_idx] = 1 - self.eps + (self.eps / self.nA)
        return policy

    def updateQ(self, Q_sa, Q_sa_next, reward):
        return Q_sa + (self.alpha * (reward + (self.gamma * Q_sa_next) - Q_sa))

    def updateQ_sarsamax(self, state, action, reward, next_state, done):
        maxQ_s_idx = np.argmax(self.Q[next_state])

        if not done:
            self.Q[state][action] = self.updateQ(self.Q[state][action], self.Q[next_state][maxQ_s_idx], reward)
        else:
            self.Q[state][action] = self.updateQ(self.Q[state][action], 0, reward)

    def updateQ_sarsa0(self, state, action, reward, next_state, done):
        next_action = self.select_action(state)
        if not done:
            self.Q[state][action] = self.updateQ(self.Q[state][action], self.Q[next_state][next_action], reward)
        else:
            self.Q[state][action] = self.updateQ(self.Q[state][action], 0, reward)

    def updateQ_exp_sarsa0(self, state, action, reward, next_state, done):
        policy = self.greedy_policy(state)
        if not done:
            self.Q[state][action] = self.updateQ(self.Q[state][action], np.dot(self.Q[next_state], policy), reward)
        else:
            self.Q[state][action] = self.updateQ(self.Q[state][action], 0, reward)

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """

        # return np.random.choice(self.nA)
        policy = self.greedy_policy(state)
        return np.random.choice(np.arange(self.nA), p=policy)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if self.q_update == 'sarsamax':
            self.updateQ_sarsamax(state, action, reward, next_state, done)
        elif self.q_update == 'sarsa0':
            self.updateQ_sarsa0(state, action, reward, next_state, done)
        elif self.q_update == 'exp_sarsa0':
            self.updateQ_exp_sarsa0(state, action, reward, next_state, done)
        else:
            raise NotImplemented('wrong parameter ' + self.q_update)
