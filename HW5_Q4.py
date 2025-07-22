# Q4.1: Accuracy when using large batch
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Check GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# Function to train the model and track metrics
def train_model(model, trainloader, optimizer, criterion, device, num_epochs=5):
    model.train()
    
    # Track metrics for each epoch
    epoch_losses = []
    epoch_accuracies = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar for training
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, data in enumerate(pbar):
            # Get the inputs and move to device
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (i + 1),
                'acc': 100. * correct / total
            })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = 100. * correct / total
        
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)
        
        print(f"Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.2f}%")
    
    return epoch_losses, epoch_accuracies

# Main function for Q4.1
def run_q4_1_experiment():
    print("\n" + "="*80)
    print(" Q4.1: Accuracy when using large batch ".center(80, '='))
    print("="*80)
    
    # Data transformations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    
    small_batch_size = 128  
    large_batch_size = 2048  
    # Create dataloaders
    trainloader_small = torch.utils.data.DataLoader(
        trainset, batch_size=small_batch_size, shuffle=True, num_workers=2
    )
    
    try:
        # Try creating a large batch dataloader
        trainloader_large = torch.utils.data.DataLoader(
            trainset, batch_size=large_batch_size, shuffle=True, num_workers=2
        )
        
        print(f"Training with two batch sizes:")
        print(f"1. Small batch size: {small_batch_size}")
        print(f"2. Large batch size: {large_batch_size}")
        
        # Using default SGD solver and hyperparameters from Lab 2
        # Train model with small batch size
        print(f"\nTraining with batch size {small_batch_size}:")
        model_small = ResNet18().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer_small = optim.SGD(model_small.parameters(), lr=0.001, momentum=0.9)
        
        losses_small, accuracies_small = train_model(
            model_small, trainloader_small, optimizer_small, criterion, device
        )
        
        # Train model with large batch size
        print(f"\nTraining with batch size {large_batch_size}:")
        model_large = ResNet18().to(device)
        optimizer_large = optim.SGD(model_large.parameters(), lr=0.001, momentum=0.9)
        
        losses_large, accuracies_large = train_model(
            model_large, trainloader_large, optimizer_large, criterion, device
        )
        
        # Plot results
        epochs = range(1, len(losses_small) + 1)
        
        plt.figure(figsize=(12, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(epochs, losses_small, 'b-', label=f'Batch Size {small_batch_size}')
        plt.plot(epochs, losses_large, 'r-', label=f'Batch Size {large_batch_size}')
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot accuracies
        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracies_small, 'b-', label=f'Batch Size {small_batch_size}')
        plt.plot(epochs, accuracies_large, 'r-', label=f'Batch Size {large_batch_size}')
        plt.title('Training Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('q4_1_accuracy_results.png')
        plt.show()
        
        # Report 5th epoch values as required in the assignment
        print("\nResults for the 5th epoch:")
        print(f"Batch size {small_batch_size}:")
        print(f"  Loss: {losses_small[-1]:.4f}")
        print(f"  Accuracy: {accuracies_small[-1]:.2f}%")
        
        print(f"Batch size {large_batch_size}:")
        print(f"  Loss: {losses_large[-1]:.4f}")
        print(f"  Accuracy: {accuracies_large[-1]:.2f}%")
        
        # Comparison and analysis
        print("\nComparison:")
        loss_diff = losses_large[-1] - losses_small[-1]
        acc_diff = accuracies_large[-1] - accuracies_small[-1]
        
        print(f"  Loss difference: {loss_diff:.4f} ({'higher' if loss_diff > 0 else 'lower'} for large batch)")
        print(f"  Accuracy difference: {acc_diff:.2f}% ({'higher' if acc_diff > 0 else 'lower'} for large batch)")
        
        return {
            'small_batch': {
                'size': small_batch_size,
                'losses': losses_small,
                'accuracies': accuracies_small
            },
            'large_batch': {
                'size': large_batch_size,
                'losses': losses_large,
                'accuracies': accuracies_large
            }
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"Batch size {large_batch_size} is too large for the available GPU memory.")
            print("Trying with a smaller batch size...")
            
            # Try with a smaller large batch size
            large_batch_size = large_batch_size // 2
            return run_q4_1_experiment()  # Recursively try with smaller batch
        else:
            raise e

# Run the experiment
q4_1_results = run_q4_1_experiment()