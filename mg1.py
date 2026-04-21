import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# --- HARDWARE DETECTION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ENHANCED BRAIN (CNN with Stability Layers) ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Conv Layer 1 + BatchNorm
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Conv Layer 2 + BatchNorm
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Dropout to prevent overfitting in small data silos
        self.dropout = nn.Dropout(0.25)
        
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(F.max_pool2d(self.conv1(x), 2)))
        x = F.relu(self.bn2(F.max_pool2d(self.conv2(x), 2)))
        x = x.view(-1, 64 * 5 * 5)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

# --- DATA PREP (Non-IID) ---
def get_non_iid_subsets(num_clients=5):
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    indices = np.arange(len(train_dataset))
    labels = train_dataset.targets.numpy()
    sorted_indices = indices[np.argsort(labels)]
    
    client_indices = np.array_split(sorted_indices, num_clients)
    return train_dataset, test_dataset, client_indices

# --- LOCAL TRAINING (With Telemetry) ---
def train_local_model(model, dataset, indices, epochs=1):
    model.to(device)
    model.train()
    # Higher-Order Optimization: Adding weight_decay for L2 Regularization
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    loader = DataLoader(Subset(dataset, indices), batch_size=32, shuffle=True)
    
    epoch_loss = 0
    for epoch in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    
    # Returning weights AND average loss for the dashboard
    avg_loss = epoch_loss / (len(loader) * epochs)
    return {k: v.cpu() for k, v in model.state_dict().items()}, avg_loss

# --- TESTING (Global Evaluation) ---
def test_model(model, test_dataset):
    model.to(device)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    avg_test_loss = test_loss / len(test_loader.dataset)
    model.cpu()
    return accuracy, avg_test_loss

# --- AGGREGATION ---
def federated_averaging(local_weights, client_data_sizes):
    """
    Weighted mean for floating-point weights. 
    Handles 'Long' tensors (like num_batches_tracked) by direct assignment.
    """
    total_data_points = sum(client_data_sizes)
    global_weights = {k: torch.zeros_like(v) for k, v in local_weights[0].items()}
    
    for i in range(len(local_weights)):
        weight_factor = client_data_sizes[i] / total_data_points
        
        for key in global_weights.keys():
            # Relational Logic: Check if the tensor can handle float math
            if global_weights[key].is_floating_point():
                global_weights[key] += local_weights[i][key] * weight_factor
            else:
                # For integer tensors (Long), just take the value from the first client
                # These are usually counters, not learned weights.
                if i == 0:
                    global_weights[key] = local_weights[i][key]
            
    return global_weights
print(f"🚀 Enhanced Logic Loaded (Device: {device})")
