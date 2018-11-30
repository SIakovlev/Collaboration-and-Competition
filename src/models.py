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

        state = states.view(-1, self.input_size)
        action = self.relu(self.fc1(state))
        action = self.relu(self.fc2(action))
        action = self.tanh(self.fc3(action))

        return action.view(-1, 2)


class Critic(nn.Module):
    """ Critic (Q value) Model."""

    def __init__(self, params):

        super(Critic, self).__init__()

        self.input_size = params['l1'][0]

        self.fc1 = nn.Linear(self.input_size, params['l1'][1])
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(params['l2'][0], params['l2'][1])
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(params['l3'][0], params['l3'][1])
        nn.init.xavier_uniform_(self.fc3.weight)
        self.fc4 = nn.Linear(params['l4'][0], params['l4'][1])
        nn.init.xavier_uniform_(self.fc4.weight)
        self.Q = nn.Linear(params['l5'][0], 1)
        nn.init.xavier_uniform_(self.Q.weight)

    def forward(self, state_action):

        state_action = state_action.view(-1, self.input_size)
        q_value = F.relu(self.fc1(state_action))
        q_value = F.relu(self.fc2(q_value))
        q_value = F.relu(self.fc3(q_value))
        q_value = F.relu(self.fc4(q_value))
        q_value = self.Q(q_value)

        return q_value
