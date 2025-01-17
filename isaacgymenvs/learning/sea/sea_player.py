import time
from rl_games.common.player import BasePlayer
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.tr_helpers import unsqueeze_obs
import gym
import torch
from torch import nn
import numpy as np


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


class Player(BasePlayer):
    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.need_estimator_data = False
        if "need_estimator_data" in self.player_config:
            self.need_estimator_data = self.player_config["need_estimator_data"]
        self.network = self.config["network"]
        self.actions_num = self.action_space.shape[0]
        self.actions_low = (
            torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        )
        self.actions_high = (
            torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        )
        self.mask = [False]

        self.normalize_input = self.config["normalize_input"]
        self.normalize_value = self.config.get("normalize_value", False)

        obs_shape = self.obs_shape
        config = {
            "actions_num": self.actions_num,
            "input_shape": obs_shape,
            "num_seqs": self.num_agents,
            "value_size": self.env_info.get("value_size", 1),
            "normalize_value": self.normalize_value,
            "normalize_input": self.normalize_input,
        }
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()  # false

    def get_action(self, obs, is_deterministic=False):
        self.player_config["is_train"] = False
        if self.has_batch_dimension == False:  # false
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)  # skip
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": obs,
            "rnn_states": self.states,
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict["mus"]
        action = res_dict["actions"]
        self.states = res_dict["rnn_states"]
        if is_deterministic:  # true
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:  # false, true when deploy
            current_action = torch.squeeze(current_action.detach())

        if self.need_estimator_data:  # false, true when debug
            self.estimator_output = res_dict["latent"]

        if self.clip_actions:  # true
            return rescale_actions(
                self.actions_low,
                self.actions_high,
                torch.clamp(current_action, -1.0, 1.0),
            )
        else:
            return current_action

    def restore(self, fn):
        # this way
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint["model"], strict=False)

        # save jit
        if False:
            import copy
            path = "/home/xander/Codes/IsaacGym/DexDribbler/isaacgymenvs/checkpoints/jit_go2_2"
            # path="/home/zdj/Quadruped/go2/dexdribbler/isaacgymenvs/checkpoints/jit_go2"
            
            running_mean_std_dict = self.model.running_mean_std.state_dict()
            for k in list(running_mean_std_dict.keys()):
                if k.startswith("running_mean_std.state_privilige"):
                    del running_mean_std_dict[k]
            torch.save(running_mean_std_dict, path+"/running_mean_std.pth")
            
            history_encoder_path = f"{path}/history_encoder.jit"
            history_head_path = f"{path}/history_head.jit"
            actor_mlp_path = f"{path}/actor_mlp.jit"
            mu_act_path = f"{path}/mu_act.jit"
            mu_path = f"{path}/mu.jit"
            
            history_encoder = copy.deepcopy(self.model.a2c_network.history_encoder).to('cpu')
            history_head = copy.deepcopy(self.model.a2c_network.history_head).to('cpu')
            actor_mlp = copy.deepcopy(self.model.a2c_network.actor_mlp).to('cpu')
            mu_act = copy.deepcopy(self.model.a2c_network.mu_act).to('cpu')
            mu = copy.deepcopy(self.model.a2c_network.mu).to('cpu')
            
            history_encoder_jit = torch.jit.script(history_encoder)
            history_head_jit = torch.jit.script(history_head)
            actor_mlp_jit = torch.jit.script(actor_mlp)
            mu_act_jit = torch.jit.script(mu_act)
            mu_jit = torch.jit.script(mu)
            
            history_encoder_jit.save(history_encoder_path)
            history_head_jit.save(history_head_path)
            actor_mlp_jit.save(actor_mlp_path)
            mu_act_jit.save(mu_act_path)
            mu_jit.save(mu_path)
            
            print("Jit has beed saved!")
            import sys
            sys.exit()

        if self.normalize_input and "running_mean_std" in checkpoint:  # false
            self.model.running_mean_std.load_state_dict(checkpoint["running_mean_std"])

        env_state = checkpoint.get("env_state", None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

    def reset(self):
        self.init_rnn()

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True
            # print('setting agent weights for selfplay')
            # self.env.create_agent(self.env.config)
            # self.env.set_weights(range(8),self.get_weights())

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        
        # # TODO: comment following lines
        i = 0
        self.max_steps = 1000
        obses_list = []
        obses_histroy_list = []
        actions_list = []
        
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obses = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obses, batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            for n in range(self.max_steps):
                if has_masks:  # false
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(obses, masks, is_deterministic)
                else:
                    action = self.get_action(obses, is_deterministic)

                if self.need_estimator_data:
                    print(self.estimator_output)

                # # TODO: comment following lines
                # i += 1
                # if i == 1:
                #     self.save_log(obses["state_obs"].squeeze(), action.squeeze(), mode="w")
                # else:
                #     self.save_log(obses["state_obs"].squeeze(), action.squeeze(), mode="a")

                obses_list.append(obses["state_obs"].squeeze().cpu().numpy())
                obses_histroy_list.append(obses["state_history"].squeeze().cpu().numpy())
                actions_list.append(action.squeeze().cpu().numpy())

                obses, r, done, info = self.env_step(self.env, action)
                cr += r
                steps += 1

                if render:
                    self.env.render(mode="human")
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[:: self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if "battle_won" in info:
                            print_game_res = True
                            game_res = info.get("battle_won", 0.5)
                        if "scores" in info:
                            print_game_res = True
                            game_res = info.get("scores", 0.5)

                    if self.print_stats:
                        cur_rewards_done = cur_rewards / done_count
                        cur_steps_done = cur_steps / done_count
                        if print_game_res:
                            print(
                                f"reward: {cur_rewards_done:.1f} steps: {cur_steps_done:.1} w: {game_res:.1}"
                            )
                        else:
                            print(
                                f"reward: {cur_rewards_done:.1f} steps: {cur_steps_done:.1f}"
                            )

                    sum_game_res += game_res
                    if batch_size // self.num_agents == 1 or games_played >= n_games:
                        break
            
            # # TODO: comment the following lines   
            # np.save('../record/obses.npy', np.array(obses_list), allow_pickle=False)
            # np.save('../record/obses_history.npy', np.array(obses_histroy_list), allow_pickle=False)
            # np.save('../record/actions.npy', np.array(actions_list), allow_pickle=False)
            # np.save('../record/targets.npy', np.array(self.env.store_target_dof_pos_list), allow_pickle=False)
            # import sys; sys.exit()
                
        print(sum_rewards)
        if print_game_res:
            print(
                "av reward:",
                sum_rewards / games_played * n_game_life,
                "av steps:",
                sum_steps / games_played * n_game_life,
                "winrate:",
                sum_game_res / games_played * n_game_life,
            )
        else:
            print(
                "av reward:",
                sum_rewards / games_played * n_game_life,
                "av steps:",
                sum_steps / games_played * n_game_life,
            )

    def save_log(self, obs, action, filename="../record/record.txt", mode="a"):
        obs = obs.cpu().numpy()
        action = action.cpu().numpy()
        sensor_info = {
            "projected_gravity": obs[0:3],
            "dof_pos": obs[3:15],
            "dof_vel": obs[15:27],
            "last_actions": obs[27:39],
            "gait_sin_indict": obs[39:43],
            "body_yaw": obs[43],
            "ball_states_p": obs[44:47],
            "command": obs[47:],
        }

        with open(filename, mode) as file:
            for sensor_name, sensor_values in sensor_info.items():
                file.write(f"{sensor_name}: {sensor_values}\n")
            file.write("-" * 20 + "\n")
            file.write(f"Action: {action}")
            file.write("-" * 80 + "\n")