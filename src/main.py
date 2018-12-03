from pprint import pprint
import json
import trainer
import matplotlib as mpl
import platform

if platform.system() == 'Darwin':
    mpl.use('TkAgg')  # Mac OS specific


def main(**kwargs):
    # Parse JSON settings file
    general_params = kwargs['general_params']

    num_of_episodes = general_params['num_of_episodes']
    mode = general_params['mode']

    t = trainer.Trainer(kwargs)
    if mode['train']:
        t.train(num_of_episodes=num_of_episodes)
    else:
        t.test(checkpoint_actor1_filename='checkpoint_actor1_2018-12-03_07-11.pth',
               checkpoint_actor2_filename='checkpoint_actor2_2018-12-03_07-11.pth',
               checkpoint_critic_filename='checkpoint_critic_2018-12-03_07-11.pth', time_span=10)


if __name__ == '__main__':

    with open('../settings.json') as settings:
        params = json.load(settings)

    main(**params)
