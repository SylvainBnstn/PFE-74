# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 16:02:11 2021

@author: Sylvain
"""


import matplotlib.pyplot as plt

import Model_DQN_update
import testwithdata
from Environment import unit_cost 
from Eps_Greedy_Policy import EpsilonGreedyPolicy
import numpy as np
import Environment

class DQN:
    
    def __init__(self):
        # Import des variables et fonctions
        self.target_net = Model_DQN_update.target_net
        self.to_tensor = Model_DQN_update.to_tensor
        self.to_tensor_long = Model_DQN_update.to_tensor_long
        self.env_step = Model_DQN_update.env_step
        self.price_grid = Model_DQN_update.price_grid
        self.profit_response = Model_DQN_update.profit_response
        self.policy = EpsilonGreedyPolicy()
        self.state_dim = Model_DQN_update.state_dim
        
    def env_initial_test_state(self,data):
        state = np.repeat(0,2*self.state_dim)
        state[0]= data
        return state

    def dqn_test(self,initial_state):
        
        state_test = initial_state
        
        # Select and perform an action
        q_values_test = self.target_net(self.to_tensor(state_test))
        action_test = self.policy.select_action(q_values_test.detach().numpy())

        next_state_test, reward_test = self.env_step(state_test, action_test)

        # Move to the next state
        state_test = next_state_test
        return reward_test,self.price_grid[action_test],state_test
        
        
    # def cumul_reward(reward_trace_test,p_test):
    # ####################################################
    # # Compute total reward 
    #     somme_algo_test = sum(reward_trace_test)
    #     price = data_test[:T,1]
    #     booked = data_test[:T,2]
    #     reward_from_data = booked * (price - unit_cost) 
    #     somme_data_test = sum(reward_from_data) 
        
    #     return somme_algo_test,somme_data_test,reward_from_data
    
    
    