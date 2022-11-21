import argparse
import inspect
import os
from pathlib import Path
import pickle
import logging
import sys
import numpy as np
from citylearn.citylearn import CityLearnEnv
from experiment import preliminary_setup

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

def simulate(schema, simulation_id, static=False, save_episode_agent=None, agent_filepath=None, recalculate_reward=None):
    # set filepaths and directories
    kwargs = preliminary_setup()
    result_directory = kwargs['result_directory']
    log_directory = kwargs['log_directory']
    log_filepath = os.path.join(log_directory, f'simulation_{simulation_id}.log')

    # set logger
    handler = logging.FileHandler(log_filepath,mode='w')
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)

    # set env
    env = CityLearnEnv(schema)

    if agent_filepath is None:
        agents = env.load_agent()
    else:
        with (open(Path(agent_filepath), 'rb')) as openfile:
            agents = pickle.load(openfile)

    for episode in range(env.schema['episodes']):
        observations = env.reset()

        while not env.done:
            # select actions
            actions = agents.select_actions(observations)

            # apply actions to env
            next_observations, reward, _, _ = env.step(actions)

            if recalculate_reward:
                env.reward_function.kwargs['electrical_storage_soc'] = np.array([b.observations['electrical_storage_soc'] for b in env.buildings])
                reward = env.reward_function.calculate()
            else:
                pass

            # update
            if not static:
                agents.add_to_buffer(observations, actions, reward, next_observations, done=env.done)
            else:
                pass

            observations = [o for o in next_observations]
            
            # print to log
            LOGGER.debug(
                f'Time step: {env.time_step}/{env.time_steps - 1},'\
                    f' Episode: {episode}/{env.schema["episodes"] - 1},'\
                        f' Actions: {actions},'\
                            f' Rewards: {reward}'
            )

        # save env
        with open(os.path.join(result_directory, f'{simulation_id}_episode_{int(episode)}.pkl'),'wb') as f:
            pickle.dump(env,f)

        # save agents
        if save_episode_agent is not None and episode == save_episode_agent - 1:
            with open(os.path.join(result_directory, f'{simulation_id}_agent_episode_{int(episode)}.pkl'),'wb') as f:
                pickle.dump(agents,f)
    else:
        pass

def main():
    parser = argparse.ArgumentParser(prog='buildsys_2022_simulate',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('schema',type=str)
    parser.add_argument('simulation_id',type=str)
    parser.add_argument('--static',action='store_true',dest='static')
    parser.add_argument('--save_episode_agent',type=int,default=None,dest='save_episode_agent')
    parser.add_argument('--agent_filepath',type=str,default=None,dest='agent_filepath')
    parser.add_argument('--recalculate_reward',action='store_true',dest='recalculate_reward')

    args = parser.parse_args()
    arg_spec = inspect.getfullargspec(simulate)
    kwargs = {key:value for (key, value) in args._get_kwargs() 
        if (key in arg_spec.args or (arg_spec.varkw is not None and key not in ['func','subcommands']))
    }
    simulate(**kwargs)

if __name__ == '__main__':
    sys.exit(main())
