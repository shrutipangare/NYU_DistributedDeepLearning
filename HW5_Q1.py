import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# Define a simple CNN model (similar to the one from Lab 2)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to measure training time for one epoch
def measure_epoch_time(dataloader, model, criterion, optimizer, device):
    model.train()
    start_time = time.time()
    
    for i, data in enumerate(dataloader, 0):
        # Get the inputs and labels
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
    
    end_time = time.time()
    return end_time - start_time

# Main function to run the experiments
def run_batch_size_experiment():
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define the transformation for CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # List of batch sizes to test
    batch_sizes = [32, 128, 512, 2048]
    
    print("Batch Size | Training Time (seconds)")
    print("---------|------------------------")
    
    for batch_size in batch_sizes:
        try:
            # Create dataloader with current batch size
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True, num_workers=2
            )
            
            # Initialize model, criterion and optimizer
            model = Net().to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            
            # First epoch (warmup)
            _ = measure_epoch_time(trainloader, model, criterion, optimizer, device)
            
            # Second epoch (measurement)
            epoch_time = measure_epoch_time(trainloader, model, criterion, optimizer, device)
            
            print(f"{batch_size:9d} | {epoch_time:.4f}")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{batch_size:9d} | Out of memory")
                break
            else:
                raise e
    
if __name__ == "__main__":
    run_batch_size_experiment()