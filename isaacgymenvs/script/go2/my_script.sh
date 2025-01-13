# play
python train.py task=Go1DribbleTraj train=Go1DribblePPOsea test=true headless=false checkpoint='./checkpoints/dribble-PID-106.pth' task.env.random_params.ball_drag.range_low=0.2 task.env.random_params.ball_drag.range_high=0.21 task.env.log_env_name=mid task.env.log_pt_name=PID

python train.py task=Go1Dribble train=Go1DribblePPOsea test=true headless=false checkpoint='./checkpoints/dribble-PID-106.pth' task.env.random_params.ball_drag.range_low=0.2 task.env.random_params.ball_drag.range_high=0.21 num_envs=1

python train.py task=Go1Dribble train=Go1DribblePPOsea test=true headless=false checkpoint='./checkpoints/last_Go2_shoulder_ep_60000_rew_26.663454.pth' task.env.random_params.ball_drag.range_low=0.2 task.env.random_params.ball_drag.range_high=0.21 num_envs=1

python train.py task=Go2Dribble train=Go2DribblePPOsea test=true headless=false ~task.env.priviledgeStates.dof_stiff ~task.env.priviledgeStates.dof_damp ~task.env.priviledgeStates.dof_calib ~task.env.priviledgeStates.payload ~task.env.priviledgeStates.com ~task.env.priviledgeStates.friction ~task.env.priviledgeStates.restitution ~task.env.priviledgeStates.ball_mass ~task.env.priviledgeStates.ball_restitution ~task.env.priviledgeStates.ball_states_v_1 ~task.env.priviledgeStates.ball_states_p_1 ~task.env.priviledgeStates.ball_states_v_2 ~task.env.priviledgeStates.ball_states_p_2 checkpoint='./checkpoints/dribble-baseline-106.pth' task.env.random_params.ball_drag.range_low=0.0 task.env.random_params.ball_drag.range_high=0.01 num_envs=1

python train.py task=Go2Dribble train=Go2DribblePPOsea test=true headless=false checkpoint='./checkpoints/dribble-PID-106.pth' task.env.random_params.ball_drag.range_low=0.2 task.env.random_params.ball_drag.range_high=0.21 num_envs=1

python train.py task=Go2Dribble train=Go2DribblePPOsea test=true headless=false checkpoint='/home/xander/Codes/IsaacGym/DexDribbler/isaacgymenvs/checkpoints/last_Go2_shoulder_ep_20000_rew_24.165459.pth' task.env.random_params.ball_drag.range_low=0.2 task.env.random_params.ball_drag.range_high=0.21 num_envs=1 sim_device='cuda:0' rl_device='cuda:0'

# train
python train.py task=Go1Dribble train=Go1DribblePPOsea experiment='Go1_reproduce' wandb_activate=True sim_device='cuda:1' rl_device='cuda:1'