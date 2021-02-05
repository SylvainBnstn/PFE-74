import torch 
import torch.nn as nn
import torch.nn.functional as F

class DeepQNetwork(nn.Module):
    ###########################################################################
    # initialize the neural network and the parameters
    def __init__(self, state_dim, action_dim, seed=2020, nb_node1=128, nb_node2=128):
        super(DeepQNetwork, self).__init__()
        # 3 layers in the neural network: 1 input, 1 hidden, 1 output layer
        self.seed = torch.manual_seed(seed)
        self.layer1 = nn.Linear(state_dim, nb_node1)
        self.layer2 = nn.Linear(nb_node1, nb_node2)
        self.layer3 = nn.Linear(nb_node2,action_dim)

    ###########################################################################
    # map states to actions
    def forward(self, state):
        # pass the state in the layers
        l1 = F.relu(self.layer1(state))
        l2 = F.relu(self.layer2(l1))
        return self.layer3(l2)
