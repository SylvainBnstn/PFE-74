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
    """
    Parameters : 
        path : file name of all the data cleaned
        gamma : discount factor between 0 and 1
        learn_rate : learning rate between 0 and 1
        train_proportion : training proportion between 0 and 1 (percentage)
        strat_min_prop : proportion of strategic customer minimum
        step_prop : proportion of strategic customer maximum
        batch_size : size of the batch/the sample of experience
        
    """
    def __init__(self, path, gamma ,learn_rate, train_proportion, strat_min_prop, step_prop, batch_size):
        # price_grid : join all the possible state under the format price, demand, date
        # price_grid_test : is a price_grid for the test part
        # proportion : train_proportion corresponding to a column index (= train_proportion * columns total)
        self.price_grid , self.price_grid_test , self.proportion = Import_data_VF.load_data(path,train_proportion,strat_min_prop, step_prop)
        
        # State dimension correspond to the number of columns of 1 state
        self.state_dim = len(self.price_grid[0])
        
        # household expenses, gas and electricity for an apartment
        self.unit_cost = 70
        
        # Maturity
        self.T = max(self.price_grid[:,2])# 12mois
        
        # Define the device on which torch and so the network will be allocated
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Definition of a transition of state also called an experience
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        
        # Define the policy and the target network
        self.policy_net = DeepQNetwork(self.state_dim, len(range(70,230))).to(self.device)
        self.target_net = DeepQNetwork(self.state_dim, len(range(70,230))).to(self.device)
        
        # Define the objects policy and memory with capacity associated
        self.policy = EpsilonGreedyPolicy()
        self.memory = ReplayMemory(100000)
        
        # Frequence of updating the target network
        self.TARGET_UPDATE = 20
        
        # Discount factor, the importance of the future with respect to the present
        self.GAMMA = gamma
        
        # Batch size, size of the sample of experiences
        self.BATCH_SIZE = batch_size
        
        # Optimizer, we choose Adam and give it the learning rate
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = learn_rate)

        
###############################################################################
    ### Training ###
    # Update the model
    def update_model(self, memory, policy_net, target_net):
        # If not enough experience to extract, get out of the function 
        if len(memory) < self.BATCH_SIZE:
            return
        
        # Get a sample of experiences
        transitions = memory.sample(self.BATCH_SIZE)
        
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.Transition(*zip(*transitions))
    
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
        
        # Convert to torch objects
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.stack(batch.reward)
        action_batch=action_batch.unsqueeze(2)
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        temp= policy_net(state_batch).view(action_batch.size(0),-1,action_batch.size(-1))
        state_action_values = temp.gather(1, action_batch)
        
        # Compute Q(s_t+1, a), the mask will help us not to take account 
        # the final state where we don't compute a Q_values
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach().max(1)[0]
        
        # Compute the expected Q values
        expected_state_action_values = reward_batch[:, 0] + (self.GAMMA * next_state_values)
    
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1).unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
        return state_action_values 
    
    # Initizalize the state of shape (T * state_dim) => state = (price, demand, date) * T
    def env_initial_state(self):
        state = np.zeros((self.T,self.state_dim))
        return state
    
    # Update one step ahead the state, next state = (price, demand, date) according the action took
    def env_step(self, state, action):
        next_state = np.zeros((self.T,self.state_dim))
        
        # Get the new state
        next_state[0,:] = [self.price_grid[action][0],self.price_grid[action][1],self.price_grid[action][2]]# Price, demand and date
        
        # Get back the historical state
        next_state[ 1:self.T , :] = state[0:self.T-1 , :]
        
        # Compute the demand and the reward
        demand_ = self.demand(next_state[0,0],next_state[0,2])
        reward = self.profit_t_d(next_state[0,0], demand_)
        
        return next_state, reward, demand_
    
    # Compute the profit given a price and a demand
    def profit_t_d(self, p_t, demand):
        # Compute the total cost took by Airbnb platform
        share_cost = 0.03*(p_t*demand + self.unit_cost)
        unique_cost = 0.14*(p_t*demand + self.unit_cost)
        tot_cost = share_cost + unique_cost
        
        # Return the profit, the exponential is used to boost the reward in the training when 
        # the seller succeeds in selling a night
        return p_t*np.exp(demand) - tot_cost
    
    # Get back the demand for a given price and date
    def demand(self, pt, date):
        # For a same price and date, we take the demand corresponding from the data
        # If there are several demand corresponding, then we take the mean
        list_of_demande = []
        for k in range(len(self.price_grid)):
            if pt == self.price_grid[k,0] and date == self.price_grid[k,2]:
                list_of_demande.append(self.price_grid[k,1])
        demand_ = int(np.mean(list_of_demande))
        return demand_
    
    # Convert numpy to tensor object of type float
    def to_tensor(self, x):
        return torch.from_numpy(np.array(x).astype(np.float32))
    
    # Convert numpy to tensor object of type long
    def to_tensor_long(self, x):
        return torch.tensor([[x]], device=self.device, dtype=torch.long)
    
    # Train loop
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
            # Initialize the environment
            state = self.env_initial_state()
            reward_trace = []
            p = []
            # For each time step t, compute a price (compute T prices in total)
            for t in range(self.T):
                # Select and perform an action
                with torch.no_grad():
                  q_values = self.policy_net(self.to_tensor(state))
                action = self.policy.select_action(q_values.detach().numpy())
                
                # Update the environment
                next_state, reward, _ = self.env_step(state, action)
                # Change the date to the right one in the training
                next_state[0,2] = t+1
        
                # Store the transition in memory
                self.memory.push(self.to_tensor(state), 
                            self.to_tensor_long(action), 
                            self.to_tensor(next_state) if t != self.T - 1 else None, # None can tell us if it's a final state or not
                            self.to_tensor([reward]))
        
                # Move to the next state
                state = next_state
                
                # Perform one step of the optimization (on the target network)
                self.update_model(self.memory, self.policy_net, self.target_net)
                
                # Append the reward and price to keep the trace in 1 episode
                reward_trace.append(reward)
                p.append(self.price_grid[action][0])
            
            # Save the reward and price to keep a trace for each episode
            return_trace.append(sum(reward_trace)) # type : a list of float
            p_trace.append(p) # type : a list of list of the prices
        
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        
                clear_output(wait = True)
                print(f'Episode {i_episode} of {num_episodes} ({i_episode/num_episodes*100:.2f}%)')

        return return_trace, p_trace


###############################################################################  
    ### Test ###
    # Initizalize the state of shape (T * state_dim): state = (price & demand, date) * T
    def env_initial_test_state(self,price, booked, date):
        state = np.zeros((self.T,self.state_dim))
        # Initialize the state at time t=0
        state[0,:] = [price, booked, date]
        return state
    
    # Compute the profit given a price and a demand
    def profit_t_d_test(self, p_t, demand):
        # Compute the total cost took by Airbnb platform
        share_cost = 0.03*(p_t*demand + self.unit_cost)
        unique_cost = 0.14*(p_t*demand + self.unit_cost)
        tot_cost = share_cost + unique_cost
        return p_t*demand - tot_cost
    
    # Update one step ahead the state, it's the same env_step function but with the profit function for test
    def env_step_test(self, state, action):
        next_state = np.zeros((self.T,self.state_dim))
        
        next_state[0,:] = [self.price_grid[action][0],self.price_grid[action][1],self.price_grid[action][2]]# Price, demand and date
        next_state[ 1:self.T , :] = state[0:self.T-1 , :]
        
        demand_ = self.demand(next_state[0,0],next_state[0,2])
        reward = self.profit_t_d_test(next_state[0,0], demand_)
        return next_state, reward, demand_
    
    # Test loop, quite the same as the trainig loop but without a for loop on the episode
    def dqn_test(self, price_grid_test):
        # Notify the layers that the target_net is not in training mode 
        # The one intraining mode is the policy_net
        self.target_net.eval()
        
        # Price and demand initialization
        price0 = price_grid_test[0,0]
        booked0 = price_grid_test[0,1]
        
        # Initialize the environment
        state_test = self.env_initial_test_state(price0, booked0,1)
        
        # Reward, price and demand trace
        reward_trace_test = [] # Reward
        p_test = [state_test[0,0]] # Price
        booked_test = [state_test[0,1]] # Demand
   
        for t in range(len(price_grid_test)): # Compute price for the period (here 12 months)
            # Select and perform an action
            q_values_test = self.target_net(self.to_tensor(state_test))
            action_test = self.policy.select_action_test(q_values_test.detach().numpy())
            
            # Update the environment
            next_state_test, reward_test, book = self.env_step_test(state_test, action_test)
            
            # Move to the next state
            state_test = next_state_test
            
            # Store the trace of reward, price and demand for one apt
            reward_trace_test.append(reward_test)
            p_test.append(self.price_grid[action_test][0])
            booked_test.append(book)
        
        return reward_trace_test, p_test, booked_test
    
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
    ### Interaction : once trained and tested, use the dqn in real interaction (with our model with differente clients) ###
    # Update one step the state, next state = price according to action took & demand that will be get later through customer
    def env_test_step(self, state, action):
        next_state = np.zeros((self.T,self.state_dim))
        
        #price_grid[action][0] because col 0 = col of price
        #demand initialized to False and will be obtained through clients action
        #date is moving one step
        next_state[0,:] = [self.price_grid[action][0], False, state[0,2]+1] 
        
        # Store hitorical states
        next_state[1:self.T,:] = state[0:self.T-1,:]      
        return next_state
        

    # For a given state, compute a price and return it with the actual state
    def dqn_interaction(self, initial_state):
        # Notify the layers that the target_net is not in training mode 
        # The one intraining mode is the policy_net
        self.target_net.eval()
        
        # Initialize the state
        state_test = initial_state
        
        # Select and perform an action
        q_values_test = self.target_net(self.to_tensor(state_test))
        action_test = self.policy.select_action_test(q_values_test.detach().numpy())

        # Update the environment
        next_state_test = self.env_test_step(state_test, action_test)

        # Move to the next state
        state_test = next_state_test
        
        # Price
        price = self.price_grid[action_test][0]
        return price, state_test
    

###############################################################################
###############################################################################
    ### Plot functions, showing the result ###
    
###############################################################################
    ### Plot training ###
    # Return for each episode
    # Mean of return and +/- 'range_std' standard deviation
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
        return plt.plot(range(T), np.array(p_trace[-(last_highlights+1):-1]).T, c = 'k', alpha = 0.5, linewidth=2) # We can change the color here the see the last sequance of price
    
    # Plot the result of training, return per episode & price per time steps per episode
    def plot_result(self, return_trace, p_trace):
        self.plot_return_trace(return_trace,"Time", "Avg reward all apt for each episode of training")
        fig = plt.figure(figsize=(16, 7))
        self.plot_price_schedules(p_trace, 5, 1, self.T)

###############################################################################
    ### Plot testing ###
    # Comparaison price generated vs price from data
    def plot_price(self, seq_price, data_test):
        fig = plt.figure(figsize=(16, 10))
        plt.plot(seq_price, label = "Price from algo", c='red', ls = "", marker='.')
        plt.plot(data_test, label = "Price from data", c='blue', ls = "", marker='.')
        plt.xticks(rotation=45)
        plt.xlabel("Time")
        plt.ylabel("Price ($)")
        plt.grid()    
        
        
    # Comparaison reward generated vs reward from data
    def plot_reward(self, cumul_reward_from_algo, cumul_reward_from_data):
        labely = "Sum of reward from algo"
        labely2 = "Sum of reward from data"
        fig = plt.figure(figsize=(16, 10))
        plt.plot(cumul_reward_from_algo, label = labely, c='red', ls = "", marker='.')
        plt.plot(cumul_reward_from_data, label = labely2, c='blue', ls = "", marker='.')
        plt.legend()
        plt.grid()

    # Plot the result of test
    def plot_result_test(self, price_grid_test):
        # Run the test and get the result
        seq_reward, seq_price, seq_booked = self.dqn_test(price_grid_test)
        
        # Plot price                               #price col
        self.plot_price(seq_price, price_grid_test[:,0])
        
        reward_from_data=[]
        for i in range(len(price_grid_test)):
            p, b = price_grid_test[i,0], price_grid_test[i,1]
            reward_from_data.append(self.profit_t_d_test(p,b))
        
        # Plot reward
        self.plot_reward(seq_reward, reward_from_data)
        
        return seq_reward, seq_price, seq_booked, reward_from_data