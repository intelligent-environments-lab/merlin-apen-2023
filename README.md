# MERLIN APEN 2023

![Methodology overview](https://ars.els-cdn.com/content/image/1-s2.0-S0306261923006876-ga1_lrg.jpg)

This repository provides the source-code to reproduce the [MERLIN: Multi-agent offline and transfer learning for occupant-centric operation of grid-interactive communities](https://www.sciencedirect.com/science/article/pii/S0306261923006876) paper. Refer to the paper for detailed explanations of the experiments desribed here as well as the general scope of this work.

## TL;DR

To reproduce the results in the paper, execute the [run.sh](run.sh) shell script. Once execution completes, run the following notebooks in order:

1. [general.ipynb](analysis/general.ipynb)
2. [rbc_validation.ipynb](analysis/rbc_validation.ipynb)
3. [hyperparameter_design_1.ipynb](analysis/hyperparameter_design_1.ipynb)
4. [reward_design.ipynb](analysis/reward_design_1.ipynb)
5. [rbc_reference_1.ipynb](analysis/rbc_reference_1.ipynb)
6. [rbc_reference_3.ipynb](analysis/rbc_reference_3.ipynb)
7. [deployment_strategy_1_0.ipynb](analysis/deployment_strategy_1_0.ipynb)
8. [deployment_strategy_1_1.ipynb](analysis/deployment_strategy_1_1.ipynb)
9. [deployment_strategy_2_0.ipynb](analysis/deployment_strategy_2_0.ipynb)
10. [deployment_strategy_3_0.ipynb](analysis/deployment_strategy_3_0.ipynb)
11. [deployment_strategy_3_1.ipynb](analysis/deployment_strategy_3_1.ipynb)
12. [deployment_strategy_3_2.ipynb](analysis/deployment_strategy_3_2.ipynb)

## Reproduction

Begin by cloning this repository:

```console
git clone https://github.com/intelligent-environments-lab/merlin-apen-2023.git
```

Then, change directory into the repository and install the Python [dependencies](requirements.txt):

```console
cd merlin-apen-2023
pip install requirements.txt
```

The following sections describe how to run each experiment (experiments should be run in order):

### Reference rule-based control design

> ❔ **NOTE**:
> This experiment is optional and can be skipped.

The first experiment is to define the reference RBC. The RBCs are defined in Run the following command to set up the work order and create input files needed for the simulation:

```console
python src/experiment.py rbc_validation set_work_order
```

The next command runs the CityLearn simulations for the RBC validation experiment:

```console
python src/experiment.py rbc_validation run_work_order
```

The simulation results are generated by running the following command:

```console
python src/experiment.py rbc_validation set_result_summary -d
```

Finally, analyze the results by running the [rbc_validation.ipynb](analysis/rbc_validation.ipynb) notebook from start to finish. 

The reference RBC selected in the paper is the [TOUPeakReductionFontanaRBC](src/agent.py#L38) RBC. However, one can use any of the other two RBCs: [SelfConsumptionFontanaRBC](src/agent.py#L9) or [TOURateOptimizationFontanaRBC](src/agent.py#L65). To set the reference RBC, replace the value of the `experiments:rbc_validation:optimal` in [setting.json](src/settings.json#L73).

### Reinforcement learning control design

> ❔ **NOTE**:
> This experiment is optional and can be skipped.

> ⏰ **NOTE**:
> This experiment takes a while to complete.

> 💾 **NOTE**:
> This experiment could use significant disk space.

The second experiment is to carry out a grid search to find best performing hyperparameters for the [SAC RL agent](src/agent.py#L89). One can use the default search grid or define a custom grid value in `experiments:hyperparameter_design:grid` in [setting.json](src/settings.json#L94). Begin by setting the work order followed by running the simulation and finally setting the result summary:

```console
python src/experiment.py hyperparameter_design_1 set_work_order
python src/experiment.py hyperparameter_design_1 run_work_order
python src/experiment.py hyperparameter_design_1 set_result_summary
```

Post-simulation analysis is done in the [hyperparameter_design_1.ipynb](analysis/hyperparameter_design_1.ipynb) notebook.

Either keep the already set hyperparameters or update the `experiments:rbc_validation:optimal` value in [setting.json](src/settings.json#L100).

### Reward design

> ❔ **NOTE**:
> This experiment is optional and can be skipped.

> ⏰ **NOTE**:
> This experiment takes a while to complete.

> 💾 **NOTE**:
> This experiment could use significant disk space.

Run this experiment to determine the best weights nad exponents for the [reward function](src/reward.py#L25). Like the previous experiments, first set then run the work order before setting the result summary:

```console
python src/experiment.py reward_design_1 set_work_order
python src/experiment.py reward_design_1 run_work_order
python src/experiment.py reward_design_1 set_result_summary -d
```

Post-simulation analysis is done in the [reward_design_1.ipynb](analysis/reward_design_1.ipynb) notebook.

Either keep the already set weights or update the `experiments:reward_design:weight` and `experiments:reward_design:exponent` values in [setting.json](src/settings.json#L88).

### RBC reference 1

> ❗️ **NOTE**:
> This experiment is NOT optional.

The RBC Reference 1 experiment simulates all buildings in CityLearn on full year data using the [selected reference RBC](#reference-rule-based-control-design). The results are used to compare the results from the [deployment strategy 1.0](#deployment-strategy-10) and [deployment Strategy 2.0](#deployment-strategy-20) experiments. Execute the following commands to run and set the results of this experiment:

```console
python src/experiment.py rbc_reference_1 set_work_order
python src/experiment.py rbc_reference_1 run_work_order
python src/experiment.py rbc_reference_1 set_result_summary -d
```

Finally, run the [rbc_reference_1.ipynb](analysis/rbc_reference_1.ipynb) to get the post-simulation analysis.

### RBC reference 3

> ❗️ **NOTE**:
> This experiment is NOT optional.

The RBC Reference 3 experiment simulates all buildings in CityLearn on part year data using the [selected reference RBC](#reference-rule-based-control-design). The results are used to compare the results from the [deployment strategy 3.1](#deployment-strategy-31) and [deployment strategy 3.2](#deployment-strategy-32) experiment. Execute the following commands to run and set the results of this experiment:

```console
python src/experiment.py rbc_reference_3 set_work_order
python src/experiment.py rbc_reference_3 run_work_order
python src/experiment.py rbc_reference_3 set_result_summary -d
```

Finally, run the [rbc_reference_3.ipynb](analysis/rbc_reference_3.ipynb) to get the post-simulation analysis.

### Deployment strategy 1.0

> ❗️ **NOTE**:
> This experiment is NOT optional.

> ⏰ **NOTE**:
> This experiment takes a while to complete.

> 💾 **NOTE**:
> This experiment could use significant disk space.

This experiment simulates all buildings in CityLearn and trains independent agents find the best-performing policies policy on full year data. The number of training episodes in defined in `train_episodes` in [settings.json](src/settings.json#L5). The final training episode is in fact used for a deterministic evaluation hence training happens in `train_episodes` - 1 episodes. Execute the following commands to run and set the results of this experiment:

```console
python src/experiment.py deployment_strategy_1_0 set_work_order
python src/experiment.py deployment_strategy_1_0 run_work_order
python src/experiment.py deployment_strategy_1_0 set_result_summary -d
```

Finally, run the [deployment_strategy_1_0.ipynb](analysis/deployment_strategy_1_0.ipynb) to get the post-simulation analysis.

### Deployment strategy 1.1

> ❗️ **NOTE**:
> This experiment is NOT optional.

This experiment uses the trained policies from [deployment strategy 1.0](#deployment-strategy-10) to evaluate the final seven months of data for comparison with [deployment strategy 3.1](#deployment-strategy-31). Execute the following commands to run and set the results of this experiment:

```console
python src/experiment.py deployment_strategy_1_1 set_work_order
python src/experiment.py deployment_strategy_1_1 run_work_order
python src/experiment.py deployment_strategy_1_1 set_result_summary -d
```

### Deployment strategy 2.0

> ❗️ **NOTE**:
> This experiment is NOT optional.

This experiment transfers the trained policies in [deployment strategy 1.0](#deployment-strategy-10) to other buildings and evaluates for a full year. Execute the following commands to run and set the results of this experiment:

```console
python src/experiment.py deployment_strategy_2_0 set_work_order
python src/experiment.py deployment_strategy_2_0 run_work_order
python src/experiment.py deployment_strategy_2_0 set_result_summary -d
```

Finally, run the [deployment_strategy_2_0.ipynb](analysis/deployment_strategy_2_0.ipynb) to get the post-simulation analysis.

### Deployment strategy 3.0

> ❗️ **NOTE**:
> This experiment is NOT optional.

> ⏰ **NOTE**:
> This experiment takes a while to complete.

> 💾 **NOTE**:
> This experiment could use significant disk space.

This experiment is similar to [deployment strategy 1.0](#deployment-strategy-10) except it trains on only the initial 5 months of data to find the best-performing policies. Execute the following commands to run and set the results of this experiment:

```console
python src/experiment.py deployment_strategy_3_0 set_work_order
python src/experiment.py deployment_strategy_3_0 run_work_order
python src/experiment.py deployment_strategy_3_0 set_result_summary -d
```

Finally, run the [deployment_strategy_3_0.ipynb](analysis/deployment_strategy_3_0.ipynb) to get the post-simulation analysis.

### Deployment strategy 3.1

> ❗️ **NOTE**:
> This experiment is NOT optional.

This experiment uses the trained policies from [deployment strategy 3.0](#deployment-strategy-30) to evaluate the final seven months of data. Execute the following commands to run and set the results of this experiment:

```console
python src/experiment.py deployment_strategy_3_1 set_work_order
python src/experiment.py deployment_strategy_3_1 run_work_order
python src/experiment.py deployment_strategy_3_1 set_result_summary -d
```

Finally, run the [deployment_strategy_3_1.ipynb](analysis/deployment_strategy_3_1.ipynb) to get the post-simulation analysis.

### Deployment strategy 3.2

> ❗️ **NOTE**:
> This experiment is NOT optional.

This experiment transfers the trained policies in [deployment strategy 3.0](#deployment-strategy-30) to other buildings and evaluates for the remainder of the year. Execute the following commands to run and set the results of this experiment:

```console
python src/experiment.py deployment_strategy_3_2 set_work_order
python src/experiment.py deployment_strategy_3_2 run_work_order
python src/experiment.py deployment_strategy_3_2 set_result_summary -d
```

Finally, run the [deployment_strategy_3_2.ipynb](analysis/deployment_strategy_3_2.ipynb) to get the post-simulation analysis.

## Citation

```bibtex
@article{NWEYE2023121323,
	title = {MERLIN: Multi-agent offline and transfer learning for occupant-centric operation of grid-interactive communities},
	journal = {Applied Energy},
	volume = {346},
	pages = {121323},
	year = {2023},
	issn = {0306-2619},
	doi = {https://doi.org/10.1016/j.apenergy.2023.121323},
	url = {https://www.sciencedirect.com/science/article/pii/S0306261923006876},
	author = {Kingsley Nweye and Siva Sankaranarayanan and Zoltan Nagy},
}
```