## obs
```python
obs{
    'state_obs': (num_envs, 49)
    'state_history': (num_envs, 49*25=1225)
    'state_privilige': (num_envs, 76)
}
```

### state_obs: 

1. projected_gravity: 3 [0:3]

2. dof_pos: 12 [3:15]

    (dof_pos - default_dof_pos) * dof_pos_scale(1.0)

    ```yaml
    defaultJointAngles: # = target angles when action = 0.0
    FL_hip_joint: 0.1 # [rad]
    RL_hip_joint: 0.1 # [rad]
    FR_hip_joint: -0.1 # [rad]
    RR_hip_joint: -0.1 # [rad]

    FL_thigh_joint: 0.8 # [rad]
    RL_thigh_joint: 1. # [rad]
    FR_thigh_joint: 0.8 # [rad]
    RR_thigh_joint: 1. # [rad]

    FL_calf_joint: -1.5 # [rad]
    RL_calf_joint: -1.5 # [rad]
    FR_calf_joint: -1.5 # [rad]
    RR_calf_joint: -1.5 # [rad]
    ```
3. dof_vel: 12 [15:27]
   
    dof_vel * dof_vel_scale(0.05)

4. last_actions: 12 [27:39]

5. gait_sin_indict: 4 [39:43]
    ```python
    self.gait_indices = torch.remainder(
        self.gait_indices + self.dt(0.02) * self.frequencies(3.), 1.0
    )

    foot_indices = [
        self.gait_indices + self.phases + self.offsets + self.bounds,
        self.gait_indices + self.offsets(0.0),
        self.gait_indices + self.bounds(0.0),
        self.gait_indices + self.phases(0.5),
    ]

    self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
    self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
    self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
    self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])
    ```
6. body_yaw: 1 [43]
    
    wrap to [-pi, pi]

7.  ball_states_p: 3 [44:47]
    
    z=0

8.  command: 2 [47:49]  [-1.5, 1.5]
    
    vx, vy: [-1, 1]

### state_history:
```python
self.history_buffer[:] = torch.cat(
    (self.history_buffer[:, self.num_obs :], obs), dim=1
)
```

> state_obs and state_history: torch.clamp(, -5.0, 5.0)



## action
```python
input_dict = {
    "is_train": False,
    "prev_actions": None,
    "obs": obs,
    "rnn_states": self.states,
}

# in class Network(ModelA2CContinuousLogStd.Network)
input_dict["obs"] = self.norm_obs(input_dict["obs"])
mu, logstd, value, states, latent = self.a2c_network(input_dict)   # get mu

# in class BaseModelNetwork(nn.Module)
from rl_games.algos_torch.running_mean_std import RunningMeanStdObs
obs_shape = {'state_history': (1225,), 'state_obs': (49,)}
running_mean_std = RunningMeanStdObs(obs_shape)

def norm_obs(self, observation):
    with torch.no_grad():
        return running_mean_std(observation)

# in class Network(network_builder.NetworkBuilder.BaseNetwork)
obs = input_dict["obs"]
state_obs = obs["state_obs"]
state_history = obs["state_history"]
encode = self.history_encoder(state_history)  # (num_envs, 256)
latent = self.history_head(encode)  # (num_envs, 76) estimator_output TODO: what is latent?
out = self.actor_mlp(torch.cat([state_obs, latent], dim=1))  # (num_envs, 128)
mu = self.mu_act(self.mu(out))

# in class Player(BasePlayer)
current_action = mu
return torch.clamp(current_action, -1.0, 1.0)
return rescale_actions(
    self.actions_low,  # tensor([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.], device='cuda:0')
    self.actions_high,  # tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], device='cuda:0')
    torch.clamp(current_action, -1.0, 1.0),
)

# in class Go1DribblerTraj(VecTask)
action_tensor = torch.clamp(actions, -self.clip_actions(-1.0), self.clip_actions(1.0))
actions_scaled = actions[:, :12] * self.action_scale(0.5)
actions_scaled[:, [0, 3, 6, 9]] *= self.hip_addtional_scale(0.5)


```
