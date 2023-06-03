# Get repository and set environment
git clone https://github.com/intelligent-environments-lab/merlin-apen-2023.git
cd merlin-apen-2023
pip install requirements.txt

# Reference rule-based control design (optional)
python src/experiment.py rbc_validation set_work_order
python src/experiment.py rbc_validation run_work_order
python src/experiment.py rbc_validation set_result_summary -d

# Reinforcement learning control design (optional)
python src/experiment.py hyperparameter_design_1 set_work_order
python src/experiment.py hyperparameter_design_1 run_work_order
python src/experiment.py hyperparameter_design_1 set_result_summary

# Reward design (optional)
python src/experiment.py reward_design_1 set_work_order
python src/experiment.py reward_design_1 run_work_order
python src/experiment.py reward_design_1 set_result_summary -d

# RBC references
python src/experiment.py rbc_reference_1 set_work_order
python src/experiment.py rbc_reference_1 run_work_order
python src/experiment.py rbc_reference_1 set_result_summary -d
python src/experiment.py rbc_reference_3 set_work_order
python src/experiment.py rbc_reference_3 run_work_order
python src/experiment.py rbc_reference_3 set_result_summary -d

# Deployment strategy 1.0
python src/experiment.py deployment_strategy_1_0 set_work_order
python src/experiment.py deployment_strategy_1_0 run_work_order
python src/experiment.py deployment_strategy_1_0 set_result_summary -d

# Deployment strategy 1.1
python src/experiment.py deployment_strategy_1_1 set_work_order
python src/experiment.py deployment_strategy_1_1 run_work_order
python src/experiment.py deployment_strategy_1_1 set_result_summary -d

# Deployment strategy 2.0
python src/experiment.py deployment_strategy_2_0 set_work_order
python src/experiment.py deployment_strategy_2_0 run_work_order
python src/experiment.py deployment_strategy_2_0 set_result_summary -d

# Deployment strategy 3.0
python src/experiment.py deployment_strategy_3_0 set_work_order
python src/experiment.py deployment_strategy_3_0 run_work_order
python src/experiment.py deployment_strategy_3_0 set_result_summary -d

# Deployment strategy 3.1
python src/experiment.py deployment_strategy_3_1 set_work_order
python src/experiment.py deployment_strategy_3_1 run_work_order
python src/experiment.py deployment_strategy_3_1 set_result_summary -d

# Deployment strategy 3.1
python src/experiment.py deployment_strategy_3_2 set_work_order
python src/experiment.py deployment_strategy_3_2 run_work_order
python src/experiment.py deployment_strategy_3_2 set_result_summary -d