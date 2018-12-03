import numpy as np
import random
import copy
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torch.nn.utils
import torch.nn.functional as F
from models import Actor, Critic
from replay_buffer import ReplayBuffer
from uo_process import UOProcess

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agents:
    def __init__(self, params):

        action_size = params['action_size']
        state_size = params['state_size']
        buf_params = params['buf_params']
        num_agents = params['num_of_agents']

        nn_params = params['nn_params']
        nn_params['nn_actor']['l1'][0] = state_size
        nn_params['nn_actor']['l3'][1] = action_size
        nn_params['nn_critic']['l1'][0] = (state_size + action_size) *  num_agents

        self.__actors_local = [Actor(nn_params['nn_actor']).to(device), Actor(nn_params['nn_actor']).to(device)]
        self.__actors_target = [Actor(nn_params['nn_actor']).to(device), Actor(nn_params['nn_actor']).to(device)]
        self.__critic_local = Critic(nn_params['nn_critic']).to(device)
        self.__critic_target = Critic(nn_params['nn_critic']).to(device)

        self.__action_size = action_size
        self.__state_size = state_size
        self.__num_agents = num_agents
        self.__memory = ReplayBuffer(buf_params)
        self.__t = 0

        self.gamma = params['gamma']
        self.learning_rate_actor = params['learning_rate_actor']
        self.learning_rate_critic = params['learning_rate_critic']
        self.tau = params['tau']

        self.__optimisers_actor = [optim.Adam(self.__actors_local[0].parameters(), self.learning_rate_actor),
                                   optim.Adam(self.__actors_local[1].parameters(), self.learning_rate_actor)]
        self.__optimiser_critic = optim.Adam(self.__critic_local.parameters(), self.learning_rate_critic)
        self.__uo_process = UOProcess(shape=(self.__num_agents, self.__action_size))
        # other parameters
        self.agent_loss = 0.0

    # Set methods
    def set_learning_rate(self, lr_actor, lr_critic):
        self.learning_rate_actor = lr_actor
        self.learning_rate_critic = lr_critic
        for n in range(self.__num_agents):
            for param_group in self.__optimisers_actor[n].param_groups:
                param_group['lr'] = lr_actor
        for param_group in self.__optimiser_critic.param_groups:
            param_group['lr'] = lr_critic

    # Get methods
    def get_actor(self):
        return self.__actors_local

    def get_critic(self):
        return self.__critic_local

    # Other methods
    def step(self, state, action, reward, next_state, done):
        # add experience to memory
        self.__memory.add(state, action, reward, next_state, done)

        if self.__memory.is_ready():
            self.__update()

    def choose_action(self, states, mode='train'):
        if mode == 'train':
            # state should be transformed to a tensor
            states = torch.from_numpy(np.array(states)).float().to(device)
            actions = np.zeros((self.__num_agents, self.__action_size))
            for i, actor in enumerate(self.__actors_local):
                state = states[i, :]
                actor.eval()
                with torch.no_grad():
                    action = actor(state)
                actor.train()
                actions[i, :] = action.cpu().numpy()
            actions += np.array(self.__uo_process.sample())
            return np.clip(actions, -1, 1)
        elif mode == 'test':
            # state should be transformed to a tensor
            states = torch.from_numpy(np.array(states)).float().to(device)
            actions = np.zeros((self.__num_agents, self.__action_size))
            for i, actor in enumerate(self.__actors_local):
                state = states[i, :]
                actor.eval()
                with torch.no_grad():
                    action = actor(state)
                actions[i, :] = action.cpu().numpy()
            actions += np.array(self.__uo_process.sample())
            return np.clip(actions, -1, 1)
        else:
            print("Invalid mode value")

    def reset(self, sigma):
        self.__uo_process.reset(sigma)

    def __update(self):

        for i in range(self.__num_agents):

            # update critic
            # ----------------------------------------------------------
            #
            states, actions, rewards, next_states, dones = self.__memory.sample()

            states_i = states[:, i, :]
            actions_i = actions[:, i, :]
            rewards_i = rewards[:, i]
            next_states_i = next_states[:, i, :]
            dones_i = dones[:, i]

            loss_fn = nn.MSELoss()
            self.__optimiser_critic.zero_grad()

            # form target
            next_states_actions = torch.cat((next_states[:, 0, :], next_states[:, 1, :],
                                             self.__actors_target[0].forward(next_states[:, 0, :]),
                                             self.__actors_target[1].forward(next_states[:, 1, :])), dim=1)
            Q_target_next = self.__critic_target.forward(next_states_actions).detach()
            targets = (rewards_i + self.gamma * Q_target_next[:, i] * (1 - dones_i))

            # form output
            states_actions = torch.cat((states[:, 0, :], states[:, 1, :],
                                        actions[:, 0, :], actions[:, 1, :]), dim=1)
            outputs = self.__critic_local.forward(states_actions)
            mean_loss_critic = loss_fn(outputs[:, i], targets)  # minus added since it's gradient ascent
            mean_loss_critic.backward()
            self.__optimiser_critic.step()

            # update actor
            # ----------------------------------------------------------
            self.__optimisers_actor[i].zero_grad()
            predicted_actions = copy.copy(actions)
            predicted_actions[:, i, :] = self.__actors_local[i](states_i)
            mean_loss_actor = - self.__critic_local.forward(torch.cat((states[:, 0, :], states[:, 1, :],
                                                                       predicted_actions[:, 0, :],
                                                                       predicted_actions[:, 1, :]), dim=1))[:, i].mean()
            mean_loss_actor.backward()
            self.__optimisers_actor[i].step()   # update actor

            self.__soft_update(self.__critic_local, self.__critic_target, self.tau)
            self.__soft_update(self.__actors_local[i], self.__actors_target[i], self.tau)

    @staticmethod
    def __soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
