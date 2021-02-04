from collections import namedtuple
import random

Transition = namedtuple(
        'Transition',
        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory():
    ###########################################################################
    # initialize memory and its capacity
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
        self.experience = namedtuple('Experience',
        ('state', 'action', 'next_state', 'reward'))
        
    ###########################################################################
    # store experience in ReplayMemory when an experience occur
    # when capacity is reached, the oldest experience is supress in favor of the new one
    def push(self, state, action, next_state, reward):
        e = self.experience(state, action, next_state, reward)
        if len(self.memory) < self.capacity:
            self.memory.append(e)
        else:
            self.memory[self.push_count % self.capacity] = e
        self.push_count += 1
    
    ###########################################################################
    # sample N random experiences, N=batch_size
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    ###########################################################################
    # return len of actual memory useful to know when it's possible to sample from memory
    def __len__(self):
        return len(self.memory)
