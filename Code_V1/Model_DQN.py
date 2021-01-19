import torch
import torch.optim as optim

import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from collections import namedtuple
import torch.nn.functional as F
import pandas as pd

from DQN_V2 import DeepQNetwork
from Replay_memory import ReplayMemory
from Eps_Greedy_Policy import EpsilonGreedyPolicy
import Import_data


class DQN:
    def __init__(self):
        # Import des variables et fonctions 
        
        self.price, self.booked= Import_data.get_data()
        
        self.price_grid = Import_data.training_data(self.price,self.booked)
        
        self.data_test = Import_data.testing_data_2019(self.price,self.booked)[0].to_numpy()
        self.data_test_booked=Import_data.testing_data_2019(self.price,self.booked)[1].to_numpy() 
        
        self.data_test = Import_data.testing_data_2020(self.price,self.booked)[0].to_numpy()
        self.data_test_booked=Import_data.testing_data_2020(self.price,self.booked)[1].to_numpy() 
        
        
        self.state_dim = len(self.price_grid)
        self.unit_cost = 20
        
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
        self.nb_price_to_compute = 100 
        

        
    def profit_t_d(self, p_t, demand):
        
        ### Commission of 14% from the price by number the booked nights
        ### total cost = cost from 
        share_cost = 0.03*(p_t*demand + self.unit_cost)
        unique_cost =  0.14*(p_t*demand + self.unit_cost)
        total_cost= share_cost+ unique_cost
        profit = demand*p_t - unique_cost
        return profit
     
    ### Visualization functions ###
    # Return for each episode
    def plot_return_trace(self,returns, labelx,labely,smoothing_window=10, range_std=2):
        plt.figure(figsize=(16, 7))
        plt.xlabel(labelx)
        plt.ylabel(labely)
        returns_df = pd.Series(returns)
        ma = returns_df.rolling(window=smoothing_window).mean()
        mstd = returns_df.rolling(window=smoothing_window).std()
        plt.plot(ma, c = 'blue', alpha = 1.00, linewidth = 1)
        plt.fill_between(mstd.index, ma-range_std*mstd, ma+range_std*mstd, color='blue', alpha=0.2)

    # Sequence of price for each episode
    def plot_price_schedules(self,p_trace, sampling_ratio, last_highlights,T, fig_number=None):
        plt.figure(fig_number);
        plt.xlabel("Time step");
        plt.ylabel("Price ($)");
        plt.plot(range(T), np.array(p_trace[0:-1:sampling_ratio]).T, c = 'k', alpha = 0.05)
        return plt.plot(range(T), np.array(p_trace[-(last_highlights+1):-1]).T, c = 'red', alpha = 0.5, linewidth=2)

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
    
    # Initizalize the state of shape (2xstate_dim)
    def env_initial_state(self):
        state = np.repeat([[0],[0]],2*self.state_dim, axis=1)
        return state
    
    # Update one step the state, next state = price & demand according the action took
    def env_step(self, state, action):
        next_state = np.repeat([[0],[0]],2*self.state_dim, axis=1)
        next_state[:,0] = self.price_grid[action][0:2]
        next_state[:, 1:self.state_dim] = state[:, 0:self.state_dim-1]
        demand = self.demand(next_state[0,0],next_state[0,1])
        reward = self.profit_t_d(next_state[0,0], demand)
        return next_state, reward, demand
    
    def demand(self, pt, pt_1):
        diff = pt - pt_1
        closer_variation = [abs(diff - diff_data) for diff_data in self.price_grid[:,1]]#colonne 2 = diff de prix from data
        closer_index = np.argmin(closer_variation)
        demand = self.price_grid[closer_index,2]# col 2 = demand
        return demand

    # Convert numpy to tensor object of type float
    def to_tensor(self, x):
        return torch.from_numpy(np.array(x).astype(np.float32))
    
    # Convert numpy to tensor object of type long
    def to_tensor_long(self, x):
        return torch.tensor([[x]], device=self.device, dtype=torch.long)
    
    # Train
    def dqn_training(self):
        # The target_net load the parmeters of the policy_net
        # state_dict() maps each layer to its parameter tensor
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Notify the layers that the target_net is not in training mode 
        # The one intraining mode is the policy_net
        self.target_net.eval()
        
        num_episodes = 200
        return_trace = []
        p_trace = [] # price schedules used in each episode
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
    
    def plot_p_d(self):
        fig = plt.figure(figsize=(16, 10))
        plt.title("Price & Demand from data train over time step (month)")
        plt.plot(self.price_grid[:,0], label = "Price", c = "red")#prix
        plt.plot(self.price_grid[:,1], label = "Demand", c= "green")#demande
        plt.legend()
        plt.grid()
    
    # Plot the result of training, return per episode & price per time steps per episode
    def plot_result(self, return_trace, p_trace):
        self.plot_p_d()
        labelx=" okok "
        labely="  nanana "
        self.plot_return_trace(return_trace,labelx,labely,10)
        fig = plt.figure(figsize=(16, 7))
        self.plot_price_schedules(p_trace, 5, 1, self.T,fig.number)
       
###############################################################################  
    ### Test ###
    # Initialize the state
    def env_initial_test_state(self,data):
        
        state = np.repeat([[0],[0]],2*self.state_dim, axis=1)
        state[:,0]= data
        return state
    
    def dqn_test(self,data_test,data_test_booked):
        # Initialization
        
        seq_price_all_apt=[]
        seq_reward_all_apt=[]
        seq_booked_all_apt=[]
        
        
        for k in range(len(data_test)):
            # take the Kth apartment 
            
            #state_test = self.env_initial_test_state(data_test[k,0],data_test_booked[k,0])
            state_test = self.env_initial_test_state(data_test[k,0])
            reward_trace_test = []
            p_test = [state_test[0,0]] # Price
            booked_test = [state_test[1,0]] # Demand
       
            for t in range(self.nb_price_to_compute): # Compute price for 3 years (12*3 prices)
                # Select and perform an action
                q_values_test = self.target_net(self.to_tensor(state_test))[0]
                action_test = self.policy.select_action_test(q_values_test.detach().numpy())
            
                next_state_test, reward_test, book = self.env_step(state_test, action_test)
                booked_test.append(book)
                
                # Move to the next state
                state_test = next_state_test
            
                reward_trace_test.append(reward_test)
                p_test.append(self.price_grid[action_test][0])
        
            seq_price_all_apt.append(p_test)
            seq_reward_all_apt.append(reward_trace_test)
            seq_booked_all_apt.append(booked_test)
          
            
        return seq_reward_all_apt, seq_price_all_apt, seq_booked_all_apt
    
    # Compute total reward
    
    def cumul_reward(self, seq_reward_all_apt,data_test,data_test_booked,year):
        
        cumul_reward_from_algo = [sum(seq_reward_all_apt[k] for k in range(len(seq_reward_all_apt)))]
        
        #price = self.data_test[:self.nb_price_to_compute]
        #booked = self.data_test[:self.nb_price_to_compute]
        price= data_test
        booked=data_test_booked
        
        cumul_r_from_data=[]
        
        for k in range(len(price)):
            reward_from_data= self.profit_t_d(price[k],booked[k])
            cumul_r_from_data.append(sum(reward_from_data))
        
        cumul_reward_from_data = [sum(cumul_r_from_data[k] for k in range(len(cumul_r_from_data)))]
        
        labelx=" year_"+year
        labely="Reward for "+year
        
        self.plot_return_trace(cumul_reward_from_data,labelx,labely)

#        print("FROM algo\t Cumulative reward for ", self.nb_price_to_compute , " months : ", cumul_reward_from_algo)
#        print("FROM data\t Cumulative reward for ", self.nb_price_to_compute , " months : ", cumul_reward_from_data)
#        print("FROM algo\t Average reward in 1 year : ", cumul_reward_from_algo*12/self.nb_price_to_compute)
#        print("FROM data\t Average reward in 1 year : ", cumul_reward_from_data*12/self.nb_price_to_compute)
#        
        return cumul_reward_from_algo, cumul_reward_from_data, reward_from_data



    # Comparaison prix généré vs prix des data
    def plot_price(self, p_test):
        fig = plt.figure(figsize=(16, 10))
        plt.title("Price generated VS Price data")
        plt.plot(p_test , label = "Algo Price ")
        plt.plot(self.data_test[:self.nb_price_to_compute,0], label = "Data Price")
        plt.legend()
        plt.grid()
        
    # Comparaison reward généré vs reward des data
    def plot_reward(self, reward_trace_test, reward_from_data):
        fig = plt.figure(figsize=(16, 10))
        plt.title("Rewards")
        plt.plot(reward_trace_test, label ="Reward per month from algo")
        plt.plot(reward_from_data, label = "Reward per month from data")
        plt.legend()
        plt.grid()
        
    # Analysis of price & demand from algo vs from data
    def analysis_price_demand(self, p_test, booked_test):
        fig = plt.figure(figsize=(16, 10))
        plt.title("Price & Demand from data over time step (month)")
        plt.plot(self.data_test[:self.nb_price_to_compute,0], label = "Price", c = "red")#prix
        plt.plot(self.data_test[:self.nb_price_to_compute,1], label = "Demand", c= "green")#demande
        plt.legend()
        plt.grid()
        
        fig = plt.figure(figsize=(16, 10))
        plt.title("Price & Demand from algo over time step (month)")
        plt.plot(p_test, label = "Price", c= "red")#prix
        plt.plot(booked_test, label = "Demand", c= "green")#demande
        plt.legend()
        plt.grid()


    def plot_result_test(self):
        seq_reward_all_apt, seq_price_all_apt, seq_booked_all_apt = self.dqn_test(self.data_test,self.data_test_booked)
        reward_from_data = self.cumul_reward(seq_reward_all_apt)[2]
        #self.plot_price(seq_price_all_apt)
        #self.plot_reward(seq_reward_all_apt,reward_from_data)
        #self.analysis_price_demand(seq_price_all_apt, seq_booked_all_apt)






###############################################################################
    ### Interaction : once trained and tested, use the dqn in real interaction ###
    # Update one step the state, next state = price according to action took & demand that will be get later through customer
    def env_test_step(self, state, action):
        next_state = np.repeat([[0],[0]],2*self.state_dim, axis=1)
        next_state[:,0] = [self.price_grid[action][0], False]
        next_state[:, 1:self.state_dim] = state[:, 0:self.state_dim-1]      
        return next_state
        

    # For a given state, compute a price and return it with the reward and the actual state
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
        ### state_test = price, bool (demand)
        
        return price, state_test
    
dqn = DQN()

def execute_train():
    return_trace, p_trace = dqn.dqn_training()
    dqn.plot_result(return_trace, p_trace)
    
    
def execute_test():
    dqn.plot_result_test()
    
    
# this function replace the demand of the customer
def get_booked(price):
    import random
    return random.randint(0,30)


def execute_interaction():
    state = dqn.env_initial_test_state([150,0])#init price 150 init demand 0
    reward_trace = [0] #reward à 0 au début
    p_trace = [state[0,0]]
    booked =[state[0,1]]
    
    for t in range(10):
        p, state= dqn.dqn_interaction(state)
        
        booking = get_booked(p)
        state[1,0] = booking
        reward = dqn.profit_t_d(state[0,0], state[1,0])

        reward_trace.append(reward)
        p_trace.append(p)
        booked.append(booking)
        
    print("PRICE: ", p_trace)
    print("REWARD: ", reward_trace)
    print("BOOKED: ", booked)
    return p_trace, reward_trace

execute_train()
execute_test()

execute_interaction()

# Look the weight of networks to check target = policy != net (a new net)
def lookatnetparam():
    net = DeepQNetwork(2*dqn.state_dim, len(dqn.price_grid)).to(dqn.device)
    for name, param in dqn.target_net.named_parameters():
        if param.requires_grad:
            print (name, param.data)
        
    for name, param in dqn.policy_net.named_parameters():
        if param.requires_grad:
            print (name, param.data)    
        
    for name, param in net.named_parameters():
        if param.requires_grad:
            print (name, param.data) 
            
#lookatnetparam()