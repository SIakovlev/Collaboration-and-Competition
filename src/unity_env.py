import logging
from unityagents import UnityEnvironment
from gym import error, spaces


class UnityGymException(error.Error):
    """
    Any error related to the gym wrapper of ml-agents.
    """
    pass


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gym_unity")


class UnityEnv:

    def __init__(self, params):

        environment_filename = params['path']
        seed = params['seed']

        self._env = UnityEnvironment(environment_filename, seed=seed)
        self._action_space_size = None
        self._current_state = None

        self.brain_name = self._env.brain_names[0]
        brain = self._env.brains[self.brain_name]

        # Check for number of agents in scene.
        env_info = self._env.reset(train_mode=True)[self.brain_name]

        # Set observation and action spaces
        self._action_space_size = brain.vector_action_space_size

        self._observation_space = env_info.vector_observations

    def reset(self, train_mode=True):
        info = self._env.reset(train_mode)[self.brain_name]
        obs, reward, done, info = self._single_step(info)
        return obs

    def step(self, actions):
        info = self._env.step(actions)[self.brain_name]
        obs, reward, done, info = self._single_step(info)
        return obs, reward, done, info

    def _single_step(self, info):
        default_observation = info.vector_observations

        return default_observation, info.rewards, info.local_done, {
            "text_observation": info.text_observations,
            "brain_info": info}

    def close(self):
        self._env.close()

    @property
    def reward_range(self):
        return -float('inf'), float('inf')

    @property
    def action_space_size(self):
        return self._action_space_size

    @property
    def observation_space(self):
        return self._observation_space
