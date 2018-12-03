import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, params):

        super(Actor, self).__init__()

        self.input_size = params['l1'][0]

        self.fc1 = nn.Linear(self.input_size, params['l1'][1])
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(params['l2'][0], params['l2'][1])
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(params['l3'][0], params['l3'][1])
        nn.init.xavier_uniform_(self.fc3.weight)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, states):

        #states = states.view(-1, self.input_size)
        action = self.relu(self.fc1(states))
        action = self.relu(self.fc2(action))
        action = self.tanh(self.fc3(action))

        return action


class Critic(nn.Module):
    """ Critic (Q value) Model."""

    def __init__(self, params):

        super(Critic, self).__init__()

        self.input_size = params['l1'][0]

        self.fc1 = nn.Linear(self.input_size, params['l1'][1])
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(params['l2'][0], params['l2'][1])
        nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3_1 = nn.Linear(params['l3'][0], params['l3'][1])
        nn.init.xavier_uniform_(self.fc3_1.weight)

        self.fc3_2 = nn.Linear(params['l3'][0], params['l3'][1])
        nn.init.xavier_uniform_(self.fc3_2.weight)

        self.Q1 = nn.Linear(params['l4'][0], 1)
        nn.init.xavier_uniform_(self.Q1.weight)
        self.Q2 = nn.Linear(params['l4'][0], 1)
        nn.init.xavier_uniform_(self.Q2.weight)

    def forward(self, state_action):

        q_value = F.relu(self.fc1(state_action))
        q_value = F.relu(self.fc2(q_value))

        q_value1 = F.relu(self.fc3_1(q_value))
        q_value1 = self.Q1(q_value1)

        q_value2 = F.relu(self.fc3_2(q_value))
        q_value2 = self.Q2(q_value2)

        return torch.cat((q_value1, q_value2), dim=1)
