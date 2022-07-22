import argparse
import concurrent.futures
import inspect
import itertools
import os
from multiprocessing import cpu_count
from pathlib import Path
import shlex
import subprocess
import sys
import pandas as pd
from citylearn.utilities import read_json, write_json

def run(experiment):
    kwargs = preliminary_setup()
    work_order_directory = kwargs['work_order_directory']
    work_order_filepath = os.path.join(work_order_directory,f'{experiment}.sh')

    with open(work_order_filepath,mode='r') as f:
        args = f.read()
    
    args = args.split('\n')
    args = [shlex.split(a) for a in args]
    settings = get_settings()
    max_workers = settings['max_workers'] if settings.get('max_workers',None) is not None else cpu_count()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        print(f'Will use {max_workers} workers for job.')
        print(f'Pooling {len(args)} jobs to run in parallel...')
        results = [executor.submit(subprocess.run,**{'args':a,'shell':False}) for a in args]
            
        for future in concurrent.futures.as_completed(results):
            try:
                print(future.result())
            except Exception as e:
                print(e)

def set_hyperparameter_design_work_order():
    experiment = 'hyperparameter_design'
    kwargs = preliminary_setup()
    schema = kwargs['schema']
    schema_directory = kwargs['schema_directory']
    src_directory = kwargs['src_directory']
    misc_directory = kwargs['misc_directory']
    work_order_directory = kwargs['work_order_directory']
    tacc_directory = kwargs['tacc_directory']
    settings = get_settings()

    # set active buildings
    train_buildings = settings['design_buildings']

    for building in schema['buildings']:
        schema['buildings'][building]['include'] = True if int(building.split('_')[-1]) in train_buildings else False

    # set reward
    schema['reward'] = settings[experiment]['reward']

    # hyperparameter definition
    hyperparameter_grid = settings[experiment]['grid']
    hyperparameter_grid['seed'] = settings['seeds']
    param_names = list(hyperparameter_grid.keys())
    param_values = list(hyperparameter_grid.values())
    param_values_grid = list(itertools.product(*param_values))
    grid = pd.DataFrame(param_values_grid,columns=param_names)
    grid = grid.sort_values(['seed'])
    grid['buildings'] = str(train_buildings)
    grid['simulation_id'] = grid.reset_index().index.map(lambda x: f'{experiment}_{x}')
    grid.to_csv(os.path.join(misc_directory,f'{experiment}_grid.csv'),index=False)

    # design work order
    work_order = []

    for i, params in enumerate(grid.to_dict('records')):
        params['seed'] = int(params['seed'])
        schema['agent']['attributes'] = {
            **schema['agent']['attributes'],
            **params
        }
        schema_filepath = os.path.join(schema_directory,f'{params["simulation_id"]}.json')
        write_json(schema_filepath, schema)
        work_order.append(f'python {os.path.join(src_directory,"simulate.py")} {schema_filepath} {params["simulation_id"]}')

    # write work order and tacc job
    work_order = '\n'.join(work_order)
    tacc_job = get_tacc_job(experiment)
    work_order_filepath = os.path.join(work_order_directory,f'{experiment}.sh')
    tacc_job_filepath = os.path.join(tacc_directory,f'{experiment}.sh')

    for d, p in zip([work_order,tacc_job],[work_order_filepath,tacc_job_filepath]):
        with open(p,'w') as f:
            f.write(d)

def set_rbc_validation_work_order():
    experiment = 'rbc_validation'
    kwargs = preliminary_setup()
    schema = kwargs['schema']
    schema_directory = kwargs['schema_directory']
    src_directory = kwargs['src_directory']
    misc_directory = kwargs['misc_directory']
    work_order_directory = kwargs['work_order_directory']
    settings = get_settings()

    # update general settings
    schema['simulation_end_time_step'] = settings["test_end_time_step"]
    schema['episodes'] = 1
    grid = pd.DataFrame({'type':settings[experiment]['type']}) 
    grid['simulation_id'] = grid.reset_index().index.map(lambda x: f'{experiment}_{x}')
    grid.to_csv(os.path.join(misc_directory,f'{experiment}_grid.csv'),index=False)
    work_order = []

    # update agent
    for i, params in enumerate(grid.to_records(index=False)):
        schema['agent'] = {
            'type': params['type'],
            'attributes': settings[experiment]['attributes']
            **{
                'hour_index':settings['observations'].index('hour'),
                'soc_index':settings['observations'].index('electrical_storage_soc'),
                'net_electricity_consumption_index':settings['observations'].index('net_electricity_consumption')
            }
        }
        schema_filepath = os.path.join(schema_directory,f'{params["simulation_id"]}.json')
        write_json(schema_filepath, schema)
        work_order.append(f'python {os.path.join(src_directory,"simulate.py")} {schema_filepath} {params["simulation_id"]}')

    # write work order
    work_order = '\n'.join(work_order)
    work_order_filepath = os.path.join(work_order_directory,f'{experiment}.sh')

    with open(work_order_filepath,'w') as f:
        f.write(work_order)

    return work_order_filepath

def set_reward_design_work_order():
    # buildings to include
    experiment = 'reward_design'
    kwargs = preliminary_setup()
    schema = kwargs['schema']
    misc_directory = kwargs['misc_directory']
    schema_directory = kwargs['schema_directory']
    src_directory = kwargs['src_directory']
    work_order_directory = kwargs['work_order_directory']
    tacc_directory = kwargs['tacc_directory']
    settings = get_settings()
    train_buildings = settings['design_buildings']
    
    for building in schema['buildings']:
        schema['buildings'][building]['include'] = True if int(building.split('_')[-1]) in train_buildings else False

    # reward definition
    reward_design_grid = settings[experiment]['grid']
    reward_design_grid['seed'] = settings['seeds']
    param_names = list(reward_design_grid.keys())
    param_values = list(reward_design_grid.values())
    param_values_grid = list(itertools.product(*param_values))
    grid = pd.DataFrame(param_values_grid,columns=param_names)
    grid = grid.sort_values(['seed','weight','exponent'])
    grid['buildings'] = str(train_buildings)
    grid['simulation_id'] = grid.reset_index().index.map(lambda x: f'{experiment}_{x}')
    grid.to_csv(os.path.join(misc_directory,f'{experiment}_grid.csv'),index=False)

    # design work order
    work_order = []

    for i, params in enumerate(grid.to_records(index=False)):
        schema['reward_function'] = {
            'type': params['type'],
            'attributes': {
                'electricity_price_weight': float(params['weight']),
                'carbon_emission_weight': float(1.0 - params['weight']),
                'electricity_price_exponent': float(params['exponent']),
                'carbon_emission_exponent': float(params['exponent']),
            }  
        }
        schema['agent']['attributes']['seed'] = int(params['seed'])
        schema_filepath = os.path.join(schema_directory,f'{params["simulation_id"]}.json')
        write_json(schema_filepath, schema)
        work_order.append(f'python {os.path.join(src_directory,"simulate.py")} {schema_filepath} {params["simulation_id"]}')

    # write work order and tacc job
    work_order = '\n'.join(work_order)
    tacc_job = get_tacc_job(experiment)
    work_order_filepath = os.path.join(work_order_directory,f'{experiment}.sh')
    tacc_job_filepath = os.path.join(tacc_directory,f'{experiment}.sh')

    for d, p in zip([work_order,tacc_job],[work_order_filepath,tacc_job_filepath]):
        with open(p,'w') as f:
            f.write(d)

def preliminary_setup():
    # set filepaths and directories
    root_directory = os.path.join(*Path(os.path.dirname(__file__)).absolute().parts[0:-1])
    src_directory = os.path.join(*Path(os.path.dirname(__file__)).absolute().parts)
    data_set_directory = os.path.join(root_directory,'data','citylearn_challenge_2022_phase_3')
    schema_directory = os.path.join(root_directory,'data','schema')
    work_order_directory = os.path.join(root_directory,'data','work_order')
    misc_directory = os.path.join(root_directory,'data','misc')
    tacc_directory = os.path.join(root_directory,'tacc')
    os.makedirs(schema_directory,exist_ok=True)
    os.makedirs(work_order_directory,exist_ok=True)
    os.makedirs(misc_directory,exist_ok=True)
    os.makedirs(tacc_directory,exist_ok=True)

    # general simulation settings
    settings = get_settings()
    schema = read_json(os.path.join(data_set_directory,'schema.json'))
    schema['simulation_start_time_step'] = settings["train_start_time_step"]
    schema['simulation_end_time_step'] = settings["train_end_time_step"]
    schema['episodes'] = settings["train_episodes"]
    schema['root_directory'] = data_set_directory
    # set active observations
    for o in schema['observations']:
        schema['observations'][o]['active'] = True if o in settings['observations'] else False

    # define agent
    schema['agent'] = settings['default_agent']

    return {
        'schema': schema, 
        'root_directory': root_directory, 
        'src_directory': src_directory, 
        'schema_directory': schema_directory, 
        'work_order_directory': work_order_directory, 
        'misc_directory': misc_directory,
        'tacc_directory': tacc_directory,
    }

def get_settings():
    src_directory = os.path.join(*Path(os.path.dirname(__file__)).absolute().parts)
    settings_filepath = os.path.join(src_directory,'settings.json')
    settings = read_json(settings_filepath)
    return settings

def get_tacc_job(experiment):
    settings = get_settings()
    queue = settings[experiment]['tacc_queue']
    nodes = settings['tacc_queue'][queue]['nodes']
    time = settings['tacc_queue'][queue]['time']
    return '\n'.join([
        '#!/bin/bash',
        f'#SBATCH -p {queue}',
        f'#SBATCH -J citylearn_buildsys_2022_{experiment}',
        f'#SBATCH -N {nodes}',
        '#SBATCH --tasks-per-node 1',
        f'#SBATCH -t {time}',
        '#SBATCH --mail-user=nweye@utexas.edu',
        '#SBATCH --mail-type=all',
        f'#SBATCH -o slurm_{experiment}.out',
        '#SBATCH -A DemandAnalysis',
        '',
        '# load modules',
        'module load launcher',
        '',
        '# activate virtual environment',
        'source /work/07083/ken658/projects/citylearn/buildsys_2022/env/bin/activate',
        '',
        '# set launcher environment variables',
        'export LAUNCHER_WORKDIR="/work/07083/ken658/projects/citylearn/buildsys_2022/data/work_order"',
        f'export LAUNCHER_JOB_FILE="{experiment}.sh"',
        '',
        '${LAUNCHER_DIR}/paramrun',
    ])

def set_work_order(experiment):
    func = {
        'reward_design':set_reward_design_work_order,
        'hyperparameter_design':set_hyperparameter_design_work_order,
        'rbc_validation':set_rbc_validation_work_order,
    }[experiment]
    func()

def get_experiments():
    return [
        'reward_design',
        'hyperparameter_design',
        'rbc_validation',
    ]

def main():
    parser = argparse.ArgumentParser(prog='buildsys_2022_simulate',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('experiment',choices=get_experiments(),type=str)
    subparsers = parser.add_subparsers(title='subcommands',required=True,dest='subcommands')
    
    # set work order
    subparser_set_work_order = subparsers.add_parser('set_work_order')
    subparser_set_work_order.set_defaults(func=set_work_order)

    # run work order
    subparser_run_work_order = subparsers.add_parser('run_work_order')
    subparser_run_work_order.set_defaults(func=run)
    
    args = parser.parse_args()
    arg_spec = inspect.getfullargspec(args.func)
    kwargs = {
        key:value for (key, value) in args._get_kwargs() 
        if (key in arg_spec.args or (arg_spec.varkw is not None and key not in ['func','subcommands']))
    }
    args.func(**kwargs)

if __name__ == '__main__':
    sys.exit(main())