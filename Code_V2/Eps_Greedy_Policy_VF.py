import random
import numpy as np

class EpsilonGreedyPolicy:
    ###########################################################################
    # initialize epsilon parameters
    def __init__(self, eps_start=0.9, eps_end=0.05, eps_decay=400):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay = eps_decay
        self.steps_done = 0
        
    ###########################################################################
    # compute the epsilon threshold 
    # choose action via exploration/exploitation (eps-greedy strategy)
    def select_action(self, q_values):
        rdm = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) \
        * np.exp(- 1. * self.steps_done / self.decay)
        self.steps_done += 1
        if rdm > eps_threshold:         # exploitation
            return np.argmax(q_values)
        else:                           # exploration
            return random.randrange(len(q_values))
    ###########################################################################
    # choose action in test phase, always via exploitation
    def select_action_test(self, q_values):
        return np.argmax(q_values)
        