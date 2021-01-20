import torch
import torch.optim as optim

import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from collections import namedtuple
import torch.nn.functional as F
import pandas as pd

from Deep_Network_VF import DeepQNetwork
from Replay_memory_VF import ReplayMemory
from Eps_Greedy_Policy_VF import EpsilonGreedyPolicy
import Import_data_VF


class DQN:
    def __init__(self):
        # Import des variables et fonctions 
        # df_price & df_booked all date
        self.price, self.booked = Import_data_VF.get_data()
        
        self.price_grid = Import_data_VF.training_data(self.price, self.booked)
        self.data_test_2019 = Import_data_VF.testing_data_2019(self.price, self.booked)[0].to_numpy()
        self.data_test_booked_2019 = Import_data_VF.testing_data_2019(self.price, self.booked)[1].to_numpy()
        self.data_test_2020 = Import_data_VF.testing_data_2020(self.price, self.booked)[0].to_numpy()
        self.data_test_booked_2020 = Import_data_VF.testing_data_2020(self.price, self.booked)[1].to_numpy()
        
        self.state_dim = len(self.price_grid)
        self.unit_cost = 50
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

        self.policy_net = DeepQNetwork(2*self.state_dim, len(self.price_grid)).to(self.device)
        self.target_net = DeepQNetwork(2*self.state_dim, len(self.price_grid)).to(self.device)
        self.policy = EpsilonGreedyPolicy()
        self.memory = ReplayMemory(10000)
        
        self.TARGET_UPDATE = 20
        self.GAMMA = 0.9
        self.BATCH_SIZE = 512
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = 0.005)
        self.T = 12 # 12mois
        
###############################################################################
    ### Training ###
    # Update the model
    def update_model(self, memory, policy_net, target_net):
        if self.BATCH_SIZE < len(memory):
            transitions = memory.sample(self.BATCH_SIZE)
            batch = self.Transition(*zip(*transitions))
        
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
            non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
            #non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
            
            state_batch = torch.stack(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.stack(batch.reward)
        
            # q-value
            state_action_values = policy_net(state_batch)[:,0].gather(1, action_batch)
        
            next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
            next_state_values[non_final_mask] = target_net(non_final_next_states)[:,0].max(1)[0].detach()
            #next_state_values = target_net(non_final_next_states).max(1)[0].detach()
    
            # Compute the expected Q values
            expected_state_action_values = reward_batch[:, 0] + (self.GAMMA * next_state_values)  
        
            # Compute Huber loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
            
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            for param in policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
    
    # Initizalize the state of shape (2xstate_dim) : state = (price, demand) * state_dim
    def env_initial_state(self):
        state = np.repeat([[0],[0]],2*self.state_dim, axis=1)
        return state
    
    # Update one step ahead the state, next state = (price, demand) according the action took
    def env_step(self, state, action):
        next_state = np.repeat([[0],[0]],2*self.state_dim, axis=1)
        next_state[:,0] = [self.price_grid[action][0],self.price_grid[action][2]]# Price & demand
        #next_state[:,0] = self.price_grid[action][0:3:2] #get the price & the demand col 0 to col 2 by step 2
        next_state[:, 1:self.state_dim] = state[:, 0:self.state_dim-1]
        demand_ = self.demand(next_state[0,0],next_state[0,1])
        reward = self.profit_t_d(next_state[0,0], demand_)
        return next_state, reward, demand_
    
    # Compute the profit given a price and a demand
    def profit_t_d(self, p_t, demand):
        # Compute the total cost took by Airbnb platform
        share_cost = 0.03*(p_t*demand + self.unit_cost)
        unique_cost = 0.14*(p_t*demand + self.unit_cost)
        tot_cost = share_cost + unique_cost
        return p_t*demand - tot_cost
    
    # Get back the demand for a given variation of price
    def demand(self, pt, pt_1):
        # For a similar variation of price, we take the demand corresponding from the data
        diff = pt - pt_1
        closer_variation = [abs(diff - diff_data) for diff_data in self.price_grid[:,1]]#colonne 2 = diff de prix from data
        closer_index = np.argmin(closer_variation)
        demand_ = self.price_grid[closer_index,2] # col 2 = demand
        return demand_

    # Convert numpy to tensor object of type float
    def to_tensor(self, x):
        return torch.from_numpy(np.array(x).astype(np.float32))
    
    # Convert numpy to tensor object of type long
    def to_tensor_long(self, x):
        return torch.tensor([[x]], device=self.device, dtype=torch.long)
    
    # Train
    def dqn_training(self,num_episodes):
        # The target_net load the parmeters of the policy_net
        # state_dict() maps each layer to its parameter tensor
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Notify the layers that the target_net is not in training mode 
        # The one intraining mode is the policy_net
        self.target_net.eval()
        
        
        return_trace = []
        p_trace = [] # Price schedules used in each episode
        # Train num_episodes times
        for i_episode in range(num_episodes):
            state = self.env_initial_state()
            reward_trace = []
            p = []
            # For each time step, compute a price (compute T prices in total)
            for t in range(self.T):
                # Select and perform an action
                with torch.no_grad():
                  q_values = self.policy_net(self.to_tensor(state))[0]
                action = self.policy.select_action(q_values.detach().numpy())
        
                next_state, reward, _ = self.env_step(state, action)
        
                # Store the transition in memory
                self.memory.push(self.to_tensor(state), 
                            self.to_tensor_long(action), 
                            self.to_tensor(next_state) if t != self.state_dim - 1 else None, 
                            self.to_tensor([reward]))
        
                # Move to the next state
                state = next_state
        
                # Perform one step of the optimization (on the target network)
                self.update_model(self.memory, self.policy_net, self.target_net)
        
                reward_trace.append(reward)
                p.append(self.price_grid[action][0])
        
            return_trace.append(sum(reward_trace))
            p_trace.append(p)
        
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        
                clear_output(wait = True)
                print(f'Episode {i_episode} of {num_episodes} ({i_episode/num_episodes*100:.2f}%)')

        return return_trace, p_trace

###############################################################################  
    ### Test ###
    # Initizalize the state of shape (2xstate_dim): state = (price & demand) * state_dim
    def env_initial_test_state(self,price, booked):
        state = np.repeat([[0],[0]],2*self.state_dim, axis=1)
        state[:,0] = [price, booked]
        return state
    
    # Test
    def dqn_test(self, data_test, data_test_booked):
        # Initialization sequences of price, reward and demand for each apt
        seq_price_all_apt=[]
        seq_reward_all_apt=[]
        seq_booked_all_apt=[]
        
        # Go through each apt
        for k in range(len(data_test)):
            
            state_test = self.env_initial_test_state(data_test[k,0], data_test_booked[k,0])
            
            # Reward, price and demand trace for one apt (here the test is for 1 year, 2019 or 2020)
            reward_trace_test = [] # Reward
            p_test = [state_test[0,0]] # Price
            booked_test = [state_test[1,0]] # Demand
       
            for t in range(len(data_test[0])): # Compute price for the period (here 1 year)
                # Select and perform an action
                q_values_test = self.target_net(self.to_tensor(state_test))[0]
                action_test = self.policy.select_action_test(q_values_test.detach().numpy())
            
                next_state_test, reward_test, book = self.env_step(state_test, action_test)
                
                # Move to the next state
                state_test = next_state_test
                
                # Store the trace of reward, price and demand for one apt
                reward_trace_test.append(reward_test)
                p_test.append(self.price_grid[action_test][0])
                booked_test.append(book)
            
            # Store the trace of sequence of reward, price and demand for all apt
            seq_price_all_apt.append(p_test)
            seq_reward_all_apt.append(reward_trace_test)
            seq_booked_all_apt.append(booked_test)
            
        return seq_reward_all_apt, seq_price_all_apt, seq_booked_all_apt
    
    # Compute total reward of all apt, over the time
    def cumul_reward(self, seq_reward_all_apt, data_test, data_test_booked):
        # Sum of reward for all apartment over the time, from algo
        cumul_reward_from_algo = [sum([row_seq_reward_all_apt[t] for row_seq_reward_all_apt in seq_reward_all_apt]) for t in range(len(seq_reward_all_apt[0]))]
        
        price = data_test
        booked = data_test_booked
        
        # Store seq of profit for each apt
        reward_from_data = []
        for k in range(len(price)):
            # Reward for one apt
            reward_from_data.append(self.profit_t_d(price[k], booked[k]))
        
        # Sum of reward for all apartment over the time, from data
        cumul_reward_from_data = [sum([row_reward_from_data[t] for row_reward_from_data in reward_from_data]) for t in range(len(reward_from_data[0]))]
        
        return cumul_reward_from_algo, cumul_reward_from_data ### plus besoin
    
###############################################################################
    ### Interaction : once trained and tested, use the dqn in real interaction ###
    # Update one step the state, next state = price according to action took & demand that will be get later through customer
    def env_test_step(self, state, action):
        next_state = np.repeat([[0],[0]],2*self.state_dim, axis=1)
        next_state[:,0] = [self.price_grid[action][0], False] #price_grid[action][0] because col 0 = col of price, and demand initialized to False
        next_state[:, 1:self.state_dim] = state[:, 0:self.state_dim-1]      
        return next_state
        

    # For a given state, compute a price and return it with the actual state
    def dqn_interaction(self, initial_state):
        self.target_net.eval()
        # Initialize the state
        state_test = initial_state
        
        # Select and perform an action
        q_values_test = self.target_net(self.to_tensor(state_test))[0]
        action_test = self.policy.select_action_test(q_values_test.detach().numpy())

        next_state_test = self.env_test_step(state_test, action_test)

        # Move to the next state
        state_test = next_state_test
        
        # Price
        price = self.price_grid[action_test][0]
        return price, state_test
    

###############################################################################
###############################################################################
    ### Plot the result ###
    
###############################################################################
    ### Plot training ###
    # Return for each episode
    def plot_return_trace(self,returns, labelx, labely, smoothing_window=10, range_std=2):
        plt.figure(figsize=(16, 7))
        plt.xlabel(labelx)
        plt.ylabel(labely)
        returns_df = pd.Series(returns)
        ma = returns_df.rolling(window=smoothing_window).mean()
        mstd = returns_df.rolling(window=smoothing_window).std()
        plt.plot(ma, c = 'blue', alpha = 1.00, linewidth = 1)
        plt.fill_between(mstd.index, ma-range_std*mstd, ma+range_std*mstd, color='blue', alpha=0.2)
        
    # Sequence of price for each episode
    def plot_price_schedules(self,p_trace, sampling_ratio, last_highlights,T):
        plt.figure(figsize=(16,7));
        plt.xlabel("Time step (month)");
        plt.ylabel("Price ($)");
        plt.plot(range(T), np.array(p_trace[0:-1:sampling_ratio]).T, c = 'k', alpha = 0.05)
        return plt.plot(range(T), np.array(p_trace[-(last_highlights+1):-1]).T, c = 'red', alpha = 0.5, linewidth=2)
    
    # Plot the result of training, return per episode & price per time steps per episode
    def plot_result(self, return_trace, p_trace):
        self.plot_return_trace(return_trace,"Time 2015 - 2018 ", "Avg reward all apt for each episode of training")
        fig = plt.figure(figsize=(16, 7))
        self.plot_price_schedules(p_trace, 5, 1, self.T)

###############################################################################
    ### Plot testing ###
    # Comparaison price generated vs price from data
    def plot_price(self, seq_price_all_apt, data_test, year):
        fig = plt.figure(figsize=(16, 10))
        plt.title("Price generated from algo in " + year)
        for price in seq_price_all_apt:
            plt.plot(price)
        plt.xticks(rotation=45)
        plt.xlabel("Time")
        plt.ylabel("Price ($)")
        plt.grid()    
        
        fig = plt.figure(figsize=(16, 10))
        plt.title("Price from data in " + year)
        for price in data_test:
            plt.plot(price)
        plt.xticks(rotation=45)
        plt.xlabel("Time")
        plt.ylabel("Price ($)")
        plt.grid()
        
    # Comparaison reward generated vs reward from data
    def plot_reward(self, cumul_reward_from_algo, cumul_reward_from_data, year):
        labely = "Sum of reward of all apart in " + year + "from algo"
        labely2 = "Sum of reward of all apart in " + year + "from data"
        fig = plt.figure(figsize=(16, 10))
        plt.plot(cumul_reward_from_algo, label = labely, c='red')
        plt.plot(cumul_reward_from_data, label = labely2, c='blue')
        plt.legend()
        plt.grid()

    # Plot the result of test, 
    def plot_result_test(self, data_test, data_test_booked, year):
        # Run the test and get the result
        seq_reward_all_apt, seq_price_all_apt, seq_booked_all_apt = self.dqn_test(data_test, data_test_booked)
        cumul_reward_from_algo, cumul_reward_from_data = self.cumul_reward(seq_reward_all_apt, data_test, data_test_booked)
        
        # Plot Price and reward
        self.plot_price(seq_price_all_apt, data_test, year)
        self.plot_reward(cumul_reward_from_algo, cumul_reward_from_data, year)
        
        return seq_reward_all_apt, seq_price_all_apt, seq_booked_all_apt
        
 
