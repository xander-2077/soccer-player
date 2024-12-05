import torch

device = 




config = {
    "actions_num": self.actions_num,
    "input_shape": obs_shape,
    "num_seqs": self.num_agents,
    "value_size": self.env_info.get("value_size", 1),
    "normalize_value": self.normalize_value,
    "normalize_input": self.normalize_input,
}
model = network.build(config)
model.to(device)
model.eval()

ckpt_path = "/home/xander/Codes/IsaacGym/DexDribbler/isaacgymenvs/checkpoints/dribble-PID-106.pth"
ckpt = torch.load(ckpt_path)

model.load_state_dict(ckpt["model"], strict=False)


for _ in range(test_steps):
    obs
    input_dict = {
        "is_train": False,
        "prev_actions": None,
        "obs": obs,
        "rnn_states": None,
    }
    with torch.no_grad():
        res_dict = model(input_dict)
    
    
    estimator_output = res_dict["latent"]
    action = res_dict["mus"]
    rescale_actions(
        self.actions_low,
        self.actions_high,
        torch.clamp(current_action, -1.0, 1.0),
    )



breakpoint()