import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import wandb

# Directory containing saved dataset files
dataset_dir = "./dataset"
model_name = "mlp.pt"
plot_dataset = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

leg_data_list = []
rms_data_list = []


# Iterate over the saved dataset files
for filename in os.listdir(dataset_dir):
    if filename.endswith(".pkl"):
        file_path = os.path.join(dataset_dir, filename)
        loaded_data = torch.load(file_path)

        # Extract the necessary data from loaded_data
        leg_data = loaded_data["leg"]
        rms_data = loaded_data["rms"]

        # Append the data to the lists
        leg_data_list.append(leg_data[:, :, :])  # 6 - 30
        rms_data_list.append(rms_data)

# Concatenate the lists to form a larger dataset
concatenated_leg_data = torch.cat(leg_data_list, dim=0)
concatenated_rms_data = torch.cat(rms_data_list, dim=0)


def min_max_normalize(data):
    min_val = data.min()
    max_val = data.max()
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


concatenated_rms_data = min_max_normalize(concatenated_rms_data)

if plot_dataset:
    # Plot leg_data
    plt.figure(figsize=(10, 5))
    plt.plot(concatenated_leg_data[:, 1, 6], label="Leg Data")  # 6 - 30
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Leg Data")
    plt.show()

    # Plot rms_data
    plt.figure(figsize=(10, 5))
    plt.plot(concatenated_rms_data, label="RMS Data")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.title("RMS Data")
    plt.show()

# Initialize WandB
# wandb.init(project="sequence-to-number")


# Define an MLP model
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.flatten = nn.Flatten()  # Flatten the input
        self.fc1 = nn.Linear(input_size, hidden_size[0])
        self.relu = nn.ReLU()
        self.hidden_fc = []
        for i in range(len(hidden_size) - 1):
            fc = nn.Linear(hidden_size[i], hidden_size[i + 1])
            self.hidden_fc.append(fc)
            self.hidden_fc.append(nn.ReLU())
        self.hidden_fc = nn.Sequential(*self.hidden_fc)
        self.fc2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc2 = nn.Linear(hidden_size[-1], output_size)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.hidden_fc(x)
        x = self.fc2(x)
        return x


# Hyperparameters
input_size = 15 * 24  # Flattened input size
hidden_size = [128, 64, 32]
output_size = 1  # Output size (single number)
learning_rate = 0.0001
batch_size = 64
num_epochs = 200

# Create the MLP model
model = MLPModel(input_size, hidden_size, output_size)
model.to(device)

# Loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load your concatenated dataset
# Assuming you have 'concatenated_leg_data' and 'concatenated_rms_data' as tensors
dataset = TensorDataset(
    concatenated_leg_data.to(device), concatenated_rms_data.to(device)
)

# Split the dataset into training and validation sets (adjust the split ratio as needed)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

# Create DataLoaders for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


train_losses = []  # To store training losses
val_losses = []  # To store validation losses


# Training loop
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    train_losses.append(loss.item())  # Append training loss
    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            outputs = model(inputs)
            val_loss += criterion(outputs, targets).item()

    val_loss /= len(val_dataloader)
    val_losses.append(val_loss)  # Append validation loss

    # Log metrics using WandB
    # wandb.log({"Train Loss": loss.item(), "Validation Loss": val_loss})

    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Validation Loss: {val_loss:.4f}"
    )

# Save the model
print("Saving the model...")
model_scripted = torch.jit.script(model)  # Export to TorchScript
model_scripted.save(os.path.join(dataset_dir, model_name))


plt.figure(figsize=(12, 6))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.show()

# After training, you can use the trained model for predictions.
with torch.no_grad():
    output_batch = model(concatenated_leg_data[-400:, :, :].to(device))

# Assuming 'output_batch' contains the model's predictions in the shape [batch_size, output_size]
# Plot the output in order

plt.figure()
plt.plot(output_batch.to("cpu").numpy(), label="Predicted Value")
plt.plot(concatenated_rms_data[-400:].to("cpu").numpy(), label="RMS Data")
plt.xlabel("Time Step")
plt.ylabel("Predicted Value")
plt.title(f"Output for Sample 1-400")
plt.legend()
plt.show()