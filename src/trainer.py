import gym
import random
import numpy as np
import torch
import os
import sys
from agents import Agents
from unity_env import UnityEnv
from collections import deque
import datetime
import logging
from pprint import pprint
from uo_process import UOProcess

if not os.path.exists('../logs'):
    os.makedirs('../logs')

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename='../logs/run_' + datetime.datetime.today().strftime('%Y-%m-%d_%H-%M') + '.log',
                    level=logging.INFO)


class Trainer:
    def __init__(self, params):

        seed = params['general_params']['seed']
        self.__set_seed(seed=seed)

        env_params = params['env_params']
        env_params['seed'] = seed
        self.env = UnityEnv(params=env_params)

        agent_params = params['agent_params']

        self.__num_of_agents = self.env.observation_space.shape[0]
        state_size = self.env.observation_space.shape[1]
        action_size = self.env.action_space_size
        agent_params['num_of_agents'] = self.__num_of_agents
        agent_params['state_size'] = state_size
        agent_params['action_size'] = action_size
        self.agents = Agents(params=agent_params)

        trainer_params = params['trainer_params']
        self.learning_rate_decay = trainer_params['learning_rate_decay']
        self.results_path = trainer_params['results_path']
        self.model_path = trainer_params['model_path']
        self.t_max = trainer_params['t_max']

        self.exploration_noise = UOProcess()

        # data gathering variables
        self.avg_rewards = []
        self.scores = []
        self.score = 0

        self.sigma = 0.5

        print("MADDPG agent.")
        print("Configuration:")
        pprint(params)
        logging.info("Configuration: {}".format(params))

    def train(self, num_of_episodes):

        logging.info("Training:")
        reward_window = deque(maxlen=100)

        for episode_i in range(1, num_of_episodes):

            states = self.env.reset()
            self.agents.reset(self.sigma)
            scores = np.zeros(self.env.observation_space.shape[0])
            total_loss = 0

            self.sigma *= 0.99

            counter = 0
            for t in range(self.t_max):

                actions = self.agents.choose_action(states)
                next_states, rewards, dones, _ = self.env.step(actions)
                self.agents.step(states, actions, rewards, next_states, dones)
                states = next_states

                # DEBUG
                # logging.info("epsiode: {}, reward: {}, counter: {}, action: {}".
                #              format(episode_i, reward, counter, action))

                total_loss += self.agents.agent_loss
                scores += rewards
                counter += 1
                if any(dones):
                    break

            reward_window.append(np.max(scores))
            self.avg_rewards.append(np.mean(np.array(reward_window)))
            print('\rEpisode {}\tCurrent Score: {:.4f}\tAverage Score: {:.4f} '
                  '\t\tTotal loss: {:.2f}\tLearning rate (actor): {:.4f}\tLearning rate (critic): {:.4f}'.
                  format(episode_i, np.max(scores), np.mean(reward_window),
                         total_loss, self.agents.learning_rate_actor, self.agents.learning_rate_critic), end="")

            logging.info('Episode {}\tCurrent Score: {:.4f}\tAverage Score (over episodes): {:.4f} '
                         '\t\tTotal loss: {:.2f}\tLearning rate (actors): {:.4f}\tLearning rate (critic): {:.4f}'.
                         format(episode_i, np.max(scores), np.mean(reward_window),
                                total_loss, self.agents.learning_rate_actor, self.agents.learning_rate_critic))

            self.agents.learning_rate_actor *= self.learning_rate_decay
            self.agents.learning_rate_critic *= self.learning_rate_decay
            self.agents.set_learning_rate(self.agents.learning_rate_actor, self.agents.learning_rate_critic)

            if episode_i % 100 == 0:

                avg_reward = np.mean(np.array(reward_window))
                print("\rEpisode: {}\tAverage total reward: {:.2f}".format(episode_i, avg_reward))

                if avg_reward >= 1.0:
                    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode_i - 100,
                                                                                                 avg_reward))
                    if not os.path.exists(self.model_path):
                        os.makedirs(self.model_path)

                    t = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')
                    torch.save(self.agents.get_actor()[0].state_dict(), self.model_path + 'checkpoint_actor1_{}.pth'.
                               format(t))
                    torch.save(self.agents.get_actor()[1].state_dict(), self.model_path + 'checkpoint_actor2_{}.pth'.
                               format(t))
                    torch.save(self.agents.get_critic().state_dict(), self.model_path + 'checkpoint_critic_{}.pth'.
                               format(t))
                    np.array(self.avg_rewards).dump(self.results_path + 'average_rewards_{}.dat'.format(t))

        t = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')
        # reward_matrix.dump(self.results_path + 'reward_matrix_new_{}.dat'.format(t))
        np.array(self.avg_rewards).dump(self.results_path + 'average_rewards_{}.dat'.format(t))

    def test(self, checkpoint_actor1_filename, checkpoint_actor2_filename, checkpoint_critic_filename, time_span=10):
        checkpoint_actor1_path = self.model_path + checkpoint_actor1_filename
        checkpoint_actor2_path = self.model_path + checkpoint_actor2_filename
        checkpoint_critic_path = self.model_path + checkpoint_critic_filename
        self.agents.get_actor()[0].load_state_dict(torch.load(checkpoint_actor1_path))
        self.agents.get_actor()[1].load_state_dict(torch.load(checkpoint_actor2_path))
        self.agents.get_critic().load_state_dict(torch.load(checkpoint_critic_path))
        for t in range(time_span):
            state = self.env.reset(train_mode=False)
            self.score = 0
            #done = False

            while True:
                action = self.agents.choose_action(state, 'test')
                state, reward, done, _ = self.env.step(action)
                self.score += np.array(np.max(reward))
                if any(done):
                    break

            print('\nFinal score:', self.score)

        self.env.close()

    @staticmethod
    def __set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
