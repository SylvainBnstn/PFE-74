import matplotlib.pyplot as plt

import Model_DQN_update
import Import_data
#from Environment import unit_cost 
from Eps_Greedy_Policy import EpsilonGreedyPolicy
import numpy as np

###############################################################################
# Import des varibles et fonctions
target_net = Model_DQN_update.target_net
to_tensor = Model_DQN_update.to_tensor
to_tensor_long = Model_DQN_update.to_tensor_long
env_step = Model_DQN_update.env_step
price_grid = Model_DQN_update.price_grid
profit_response = Model_DQN_update.profit_response

policy = EpsilonGreedyPolicy()
state_dim = Model_DQN_update.state_dim

data_test = Import_data.get_data()[1].to_numpy()
data_train = Import_data.get_data()[0].to_numpy()

#T= len(data_test)
T = 100 # number of month limited manually to avoid too long test...
unit_cost = 50

def env_initial_test_state(data):
    state = np.repeat([[0],[0]],2*state_dim, axis=1)
    #state = np.repeat(0,2*state_dim)
    state[:,0]= data[0,1:3] #1:3 pour récup colonne 1 et 2 = le prix et la demande associé à ce prix
    return state

def DQN_test():
    # Initialization
    state_test = env_initial_test_state(data_test)
    reward_trace_test = []
    p_test = [state_test[0,0]] #price
    booked_test = [state_test[1,0]] #demand

###############################################################################
# Test
    for t in range(T):
        # Select and perform an action
        q_values_test = target_net(to_tensor(state_test))[0]
        action_test = policy.select_action(q_values_test.detach().numpy())
    
        next_state_test, reward_test = env_step( state_test, action_test)
        
        booked_test.append(next_state_test[1,0])
        # Move to the next state
        state_test = next_state_test
    
        reward_trace_test.append(reward_test)
        p_test.append(price_grid[action_test][0])
      
    return reward_trace_test, p_test, booked_test

###############################################################################
# Compute total reward 
def cumul_reward(reward_trace_test):
    somme_algo_test = sum(reward_trace_test)
    
    price = data_test[:T,1]
    booked = data_test[:T,2]
    reward_from_data = booked * (price - unit_cost)
    
    somme_data_test = sum(reward_from_data)
    
    print("FROM algo\t Cumulative reward for ", T , " months : ", somme_algo_test)
    print("FROM data\t Cumulative reward for ", T , " months : ", somme_data_test)
    print("FROM algo\t Average reward in 1 year : ", somme_algo_test*12/T)
    print("FROM data\t Average reward in 1 year : ", somme_data_test*12/T)
    
    return somme_algo_test, somme_data_test, reward_from_data

###############################################################################
#comparaison prix généré vs prix des data
def plot_price(p_test):
    fig = plt.figure(figsize=(16, 10))
    plt.title("Price generated VS Price data")
    plt.plot(p_test , label = " Price ")
    plt.plot(data_test[:T,1], label = "Data Price")
    plt.legend()
    plt.grid()
    
#comparaison reward généré vs prix des data
def plot_reward(reward_trace_test, reward_from_data):
    fig = plt.figure(figsize=(16, 10))
    plt.title("Rewards")
    plt.plot(reward_trace_test, label ="Reward per month from algo")
    plt.plot(reward_from_data, label = "Reward per month from data")
    plt.legend()
    plt.grid()
    

reward_trace_test, p_test, booked_test = DQN_test()
reward_from_data = cumul_reward(reward_trace_test)[2]
plot_price(p_test)
plot_reward(reward_trace_test,reward_from_data)


###############################################################################
# Analysis of price & demand from algo vs data
def analysis_price_demand(p_test, booked_test):
    """
    import Environment
    d_0 = Environment.d_0
    k = Environment.k
    a = Environment.a_q
    b = Environment.b_q
    d_t = Environment.d_t
    
    dt=[15]#demande init à 15 (sachant max=30)
    for i in range(1,len(p_test)):
        dt.append(d_t(p_test[i],p_test[i-1], d_0, k, a, b))
    """
    
    fig = plt.figure(figsize=(16, 10))
    plt.title("Price & Demand from data over time step (month)")
    plt.plot(data_test[:T,1], label = "Price", c = "red")#prix
    plt.plot(data_test[:T,2], label = "Demand", c= "green")#demande
    plt.legend()
    plt.grid()
    
    fig = plt.figure(figsize=(16, 10))
    plt.title("Price & Demand from algo over time step (month)")
    plt.plot(p_test, label = "Price", c= "red")#prix
    plt.plot(booked_test, label = "Demand", c= "green")#demande
    plt.legend()
    plt.grid()
    
    fig = plt.figure(figsize=(16, 10))
    plt.title("Price & Demand from data over time step (month)")
    plt.plot(data_train[:,1], label = "Price", c = "red")#prix
    plt.plot(data_train[:,2], label = "Demand", c= "green")#demande
    plt.legend()
    plt.grid()
    
    
analysis_price_demand(p_test, booked_test)
    
    




