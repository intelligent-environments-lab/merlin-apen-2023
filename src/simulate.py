import argparse
import inspect
import os
import pickle
import logging
import sys
from citylearn.citylearn import CityLearnEnv
from src.experiment import preliminary_setup

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

def simulate(schema, simulation_id, static=False):
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
    agents = env.load_agents()

    for episode in range(env.schema['episodes']):
        observations_list = env.reset()

        while not env.done:
            # select actions
            actions_list = [a.select_actions(o) for a, o in zip(agents, observations_list)]

            # apply actions to env
            next_observations_list, reward_list, _, _ = env.step(actions_list)

            # update
            if not static:
                for agent, o, a, r, n in zip(agents, observations_list, actions_list, reward_list, next_observations_list):
                    agent.add_to_buffer(o, a, r, n, done=env.done)
            else:
                pass

            observations_list = [o for o in next_observations_list]
            
            # print to log
            LOGGER.debug(
                f'Time step: {env.time_step}/{env.time_steps - 1},'\
                    f' Episode: {episode}/{env.schema["episodes"] - 1},'\
                        f' Actions: {actions_list},'\
                            f' Rewards: {reward_list}'
            )

        # save result
        with open(os.path.join(result_directory, f'{simulation_id}_episode_{int(episode)}.pkl'),'wb') as f:
            pickle.dump(env,f)

def main():
    parser = argparse.ArgumentParser(prog='buildsys_2022_simulate',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('schema',type=str)
    parser.add_argument('simulation_id',type=str)
    parser.add_argument('--static',action='store_true',dest='static')

    args = parser.parse_args()
    arg_spec = inspect.getfullargspec(simulate)
    kwargs = {key:value for (key, value) in args._get_kwargs() 
        if (key in arg_spec.args or (arg_spec.varkw is not None and key not in ['func','subcommands']))
    }
    simulate(**kwargs)

if __name__ == '__main__':
    sys.exit(main())
