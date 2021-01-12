import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
from collections import namedtuple
import torch.nn.functional as F
import pandas as pd

import torch
import torch.optim as optim

from DQN_V2 import DeepQNetwork
from Replay_memory import ReplayMemory
from Eps_Greedy_Policy import EpsilonGreedyPolicy
import Environment 

# Visualization functions
def plot_return_trace(returns, smoothing_window=10, range_std=2):
  plt.figure(figsize=(16, 7))
  plt.xlabel("Episode")
  plt.ylabel("Return ($)")
  returns_df = pd.Series(returns)
  ma = returns_df.rolling(window=smoothing_window).mean()
  mstd = returns_df.rolling(window=smoothing_window).std()
  plt.plot(ma, c = 'blue', alpha = 1.00, linewidth = 1)
  plt.fill_between(mstd.index, ma-range_std*mstd, ma+range_std*mstd, color='blue', alpha=0.2)

def plot_price_schedules(p_trace, sampling_ratio, last_highlights,T, fig_number=None):
  plt.figure(fig_number);
  plt.xlabel("Time step");
  plt.ylabel("Price ($)");
  plt.plot(range(T), np.array(p_trace[0:-1:sampling_ratio]).T, c = 'k', alpha = 0.05)
  return plt.plot(range(T), np.array(p_trace[-(last_highlights+1):-1]).T, c = 'red', alpha = 0.5, linewidth=2)


###############################################################################
# Update the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

GAMMA = 0.9
TARGET_UPDATE = 20
BATCH_SIZE = 512

import testwithdata

data_train = testwithdata.get_data()[1].to_numpy()
#T = len(data_train)
T = 12 # 12mois

price_grid = Environment.price_grid
profit_t_response = Environment.profit_t_response
profit_response = Environment.profit_response
state_dim = Environment.state_dim

def update_model(memory, policy_net, target_net):
    if BATCH_SIZE < len(memory):
            
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
    
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
        #non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])
        
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.stack(batch.reward)
    
        #q-value
        state_action_values = policy_net(state_batch).gather(1, action_batch)
    
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        #next_state_values = target_net(non_final_next_states).max(1)[0].detach()
    
        # Compute the expected Q values
        expected_state_action_values = reward_batch[:, 0] + (GAMMA * next_state_values)  
    
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
    
def env_initial_state():
    state = np.repeat(0,2*state_dim)
    return state

def env_step(t, state, action):
    next_state = np.repeat(0,2*state_dim)
    next_state[0] = price_grid[action]
    next_state[1:state_dim] = state[0:state_dim-1]
    reward = profit_t_response(next_state[0], next_state[1])
    return next_state, reward

def to_tensor(x):
  return torch.from_numpy(np.array(x).astype(np.float32))

def to_tensor_long(x):
  return torch.tensor([[x]], device=device, dtype=torch.long)


###############################################################################
# Training
policy_net = DeepQNetwork(2*state_dim, len(price_grid)).to(device)
target_net = DeepQNetwork(2*state_dim, len(price_grid)).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr = 0.005)
policy = EpsilonGreedyPolicy()
memory = ReplayMemory(10000)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

def training():
        
    num_episodes = 100
    return_trace = []
    p_trace = [] # price schedules used in each episode
    for i_episode in range(num_episodes):
        state = env_initial_state()
        reward_trace = []
        p = []
        for t in range(T):
            # Select and perform an action
            with torch.no_grad():
              q_values = policy_net(to_tensor(state))
            action = policy.select_action(q_values.detach().numpy())
    
            next_state, reward = env_step(t, state, action)
    
            # Store the transition in memory
            memory.push(to_tensor(state), 
                        to_tensor_long(action), 
                        to_tensor(next_state) if t != state_dim - 1 else None, 
                        to_tensor([reward]))
    
            # Move to the next state
            state = next_state
    
            # Perform one step of the optimization (on the target network)
            update_model(memory, policy_net, target_net)
    
            reward_trace.append(reward)
            p.append(price_grid[action])
    
        return_trace.append(sum(reward_trace))
        p_trace.append(p)
    
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    
            clear_output(wait = True)
            print(f'Episode {i_episode} of {num_episodes} ({i_episode/num_episodes*100:.2f}%)')
    
    plot_return_trace(return_trace)
    fig = plt.figure(figsize=(16, 7))
    plot_price_schedules(p_trace, 5, 1, T,fig.number)
    
    return return_trace, p_trace


return_trace, p_trace = training()


for profit in sorted(profit_response(s) for s in p_trace)[-10:]:
    print(f'Best profit results: {profit}')
