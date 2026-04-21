import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# --- PHASE 1: HARDWARE DETECTION ---
# This ensures your Victus 15 uses its NVIDIA GPU instead of the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PHASE 2: THE BRAIN (CNN Architecture) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Relational anchor to detect basic features
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10) # 10 digits (0-9)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# --- PHASE 3: THE LOGIC (Weighted Averaging) ---
def federated_averaging(local_weights, client_data_sizes):
    """
    Weighted mean based on data volume to ensure 'bigger' clients 
    have a stronger relational influence on the global model.
    """
    total_data_points = sum(client_data_sizes)
    global_weights = {k: torch.zeros_like(v) for k, v in local_weights[0].items()}
    
    for i in range(len(local_weights)):
        weight_factor = client_data_sizes[i] / total_data_points
        for key in global_weights.keys():
            global_weights[key] += local_weights[i][key] * weight_factor
            
    return global_weights

# --- PHASE 4: THE DATA (Non-IID Splitting) ---
def get_non_iid_subsets(num_clients=5):
    # Standard MNIST with normalization
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    
    # Sort by labels to create 'Bias Silos' (The Research Hook)
    indices = np.arange(len(train_dataset))
    labels = train_dataset.targets.numpy()
    sorted_indices = indices[np.argsort(labels)]
    
    client_indices = np.array_split(sorted_indices, num_clients)
    return train_dataset, client_indices

# --- PHASE 5: LOCAL TRAINING (GPU Optimized) ---
def train_local_model(model, dataset, indices, epochs=1):
    model.to(device) # Move brain to GPU
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    loader = DataLoader(Subset(dataset, indices), batch_size=32, shuffle=True)
    
    for epoch in range(epochs):
        for data, target in loader:
            # Move data tensors to the same GPU device
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    
    # Return weights back to CPU for aggregation
    return {k: v.cpu() for k, v in model.state_dict().items()}

# --- PHASE 6: TESTING (Global Evaluation) ---
def test_model(model):
    model.to(device)
    test_dataset = datasets.MNIST('./data', train=False, download=True, 
                                  transform=transforms.Compose([
                                      transforms.ToTensor(), 
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))
    test_loader = DataLoader(test_dataset, batch_size=1000)
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    model.cpu() # Release VRAM
    return accuracy

print(f"🚀 Logic Loaded (Running on: {device})")
