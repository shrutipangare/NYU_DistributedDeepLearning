# High Performance Machine Learning - HW5

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
num_gpus = torch.cuda.device_count()

print(f"Using device: {device}")
print(f"Number of available GPUs: {num_gpus}")
if torch.cuda.is_available():
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
print()

# Define the ResNet model (used for all experiments)
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

# Calculate model size (for bandwidth calculations)
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_size_bytes = total_params * 4  # assuming float32 (4 bytes)
    return total_params, total_size_bytes

# Load CIFAR-10 dataset (common for all experiments)
def load_cifar10():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    return trainset

# Common batch sizes for all experiments
batch_sizes = [32, 128, 512, 2048]  # Will try increasing up to GPU memory limit


#############################################
# Q2: Speedup Measurement
#############################################

def run_q2_speedup_experiment(trainset, q1_results):
    print("\n" + "="*80)
    print(" Q2: Speedup Measurement ".center(80, '='))
    print("="*80)
    
    # For Q2, we'll use the batch sizes that worked in Q1
    valid_batch_sizes = [bs for bs, _ in q1_results]
    
    # If there are no valid batch sizes from Q1 or we don't have multiple GPUs
    if not valid_batch_sizes or num_gpus < 2:
        print("\nCannot run Q2 experiment:")
        if not valid_batch_sizes:
            print("- No valid batch sizes from Q1")
        if num_gpus < 2:
            print(f"- Need at least 2 GPUs, but only {num_gpus} available")
        
        # Create empty results with the 1-GPU measurements from Q1
        results = {}
        for batch_size, time_1gpu in q1_results:
            results[batch_size] = {1: time_1gpu}
        
        return results
    
    results = {}
    gpu_counts = [1]
    
    # Add available GPU counts
    if num_gpus >= 2:
        gpu_counts.append(2)
    if num_gpus >= 4:
        gpu_counts.append(4)
    
    print("\nBatch Size | GPU Count | Time (s) | Speedup")
    print("-" * 45)
    
    # Run experiments for each batch size and GPU count
    for batch_size, time_1gpu in q1_results:
        results[batch_size] = {1: time_1gpu}
        
        # Report single GPU results (baseline)
        print(f"{batch_size:9d} | {1:9d} | {time_1gpu:7.4f} | {1.0:7.2f}")
        
        # Run on multiple GPUs if available
        for gpu_count in gpu_counts:
            if gpu_count > 1:
                try:
                    # Create a larger batch for multi-GPU
                    multi_gpu_batch_size = batch_size * gpu_count
                    trainloader = torch.utils.data.DataLoader(
                        trainset, batch_size=multi_gpu_batch_size, shuffle=True, num_workers=2
                    )
                    
                    # Create model and wrap with DataParallel
                    model = ResNet18()
                    model = nn.DataParallel(model, device_ids=list(range(gpu_count)))
                    model = model.to(device)
                    
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
                    
                    # First epoch (warmup)
                    print(f"Running warmup epoch with batch size {batch_size} per GPU on {gpu_count} GPUs...")
                    _ = measure_epoch_time(trainloader, model, criterion, optimizer, device)
                    
                    # Second epoch (measurement)
                    print(f"Measuring second epoch with batch size {batch_size} per GPU on {gpu_count} GPUs...")
                    multi_gpu_time = measure_epoch_time(trainloader, model, criterion, optimizer, device)
                    
                    # Store result
                    results[batch_size][gpu_count] = multi_gpu_time
                    
                    # Calculate speedup
                    speedup = time_1gpu / multi_gpu_time
                    print(f"{batch_size:9d} | {gpu_count:9d} | {multi_gpu_time:7.4f} | {speedup:7.2f}")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"{batch_size:9d} | {gpu_count:9d} | Out of memory | --")
                    else:
                        raise e
    
    # Create summary table for the report (Table 1)
    print("\nTable 1: Speedup Measurement for different Batch Size")
    print("=" * 80)
    print("           | Batch-size 32 per GPU     | Batch-size 128 per GPU    | Batch-size 512 per GPU")
    print("           | Time(sec)  | Speedup      | Time(sec)  | Speedup      | Time(sec)  | Speedup")
    print("-" * 80)
    
    # 1-GPU row
    print("1-GPU      |", end="")
    for batch_size in [32, 128, 512]:
        if batch_size in results:
            time_1gpu = results[batch_size][1]
            print(f" {time_1gpu:9.4f} | {1.0:11.2f} |", end="")
        else:
            print(" --------- | ----------- |", end="")
    print()
    
    # 2-GPU row
    if 2 in gpu_counts:
        print("2-GPU      |", end="")
        for batch_size in [32, 128, 512]:
            if batch_size in results and 2 in results[batch_size]:
                time_2gpu = results[batch_size][2]
                speedup = results[batch_size][1] / time_2gpu
                print(f" {time_2gpu:9.4f} | {speedup:11.2f} |", end="")
            else:
                print(" --------- | ----------- |", end="")
        print()
    
    # 4-GPU row
    if 4 in gpu_counts:
        print("4-GPU      |", end="")
        for batch_size in [32, 128, 512]:
            if batch_size in results and 4 in results[batch_size]:
                time_4gpu = results[batch_size][4]
                speedup = results[batch_size][1] / time_4gpu
                print(f" {time_4gpu:9.4f} | {speedup:11.2f} |", end="")
            else:
                print(" --------- | ----------- |", end="")
        print()
    
    # Visualize speedup
    if num_gpus >= 2:
        plt.figure(figsize=(10, 6))
        
        for batch_size in results:
            speedups = []
            x_values = []
            
            for gpu_count in sorted(results[batch_size].keys()):
                if gpu_count > 0:  # Skip any zero entries
                    x_values.append(gpu_count)
                    speedup = results[batch_size][1] / results[batch_size][gpu_count]
                    speedups.append(speedup)
            
            plt.plot(x_values, speedups, 'o-', label=f'Batch Size {batch_size}')
        
        # Add ideal speedup line
        max_gpus = max(gpu_counts)
        plt.plot([1, max_gpus], [1, max_gpus], 'k--', label='Ideal Linear Speedup')
        
        plt.xlabel('Number of GPUs')
        plt.ylabel('Speedup')
        plt.title('Q2: Speedup vs Number of GPUs')
        plt.grid(True)
        plt.legend()
        plt.savefig('q2_speedup_results.png')
        plt.show()
    
    
    return results

#############################################
# Q3: Computation vs Communication
#############################################

def run_q3_compute_vs_communication(trainset, q1_results, q2_results):
    print("\n" + "="*80)
    print(" Q3: Computation vs Communication ".center(80, '='))
    print("="*80)
    
    # Q3.1: Time spent in computation and communication
    print("\nQ3.1: How much time spent in computation and communication")
    print("-" * 80)
    
    # Initialize model and get parameter count for bandwidth calculations
    model = ResNet18()
    num_params, model_size_bytes = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    print(f"Model size: {model_size_bytes / (1024**2):.2f} MB")
    
    # If there are no valid batch sizes from Q1 or we don't have multiple GPUs
    if not q2_results or num_gpus < 2:
        print("\nCannot run Q3 experiment:")
        if not q2_results:
            print("- No valid results from Q2")
        if num_gpus < 2:
            print(f"- Need at least 2 GPUs, but only {num_gpus} available")
        
        # Create estimated results based on theoretical models
        print("\nEstimating results based on theoretical models:")
        
        # Calculate theoretical communication time
        def estimate_comm_time(batch_size, gpu_count):
            # Simplified model for communication time
            # T_comm = 2(n-1)/n * model_size / bandwidth
            model_size_gb = model_size_bytes / (1024**3)
            bandwidth_gbs = 20  # Assume 20 GB/s for NVLink or similar
            return 2 * (gpu_count - 1) / gpu_count * model_size_gb / bandwidth_gbs
        
        # Use the 1-GPU computation times from Q1 as baseline
        compute_times = {bs: time for bs, time in q1_results}
        
        q3_results = {"compute_times": {}, "comm_times": {}}
        
        for batch_size, time_1gpu in q1_results:
            q3_results["compute_times"][(batch_size, 1)] = time_1gpu
            q3_results["comm_times"][(batch_size, 1)] = 0
            
            # Estimate for 2 GPUs
            q3_results["compute_times"][(batch_size, 2)] = time_1gpu
            q3_results["comm_times"][(batch_size, 2)] = estimate_comm_time(batch_size, 2)
            
            # Estimate for 4 GPUs
            q3_results["compute_times"][(batch_size, 4)] = time_1gpu
            q3_results["comm_times"][(batch_size, 4)] = estimate_comm_time(batch_size, 4)
        
        # Create Table 2
        print("\nTable 2: Compute and Communication time for different Batch Size (ESTIMATED)")
        print("=" * 100)
        print("         | Batch-size 32 per GPU     | Batch-size 128 per GPU    | Batch-size 512 per GPU")
        print("         | Compute(s) | Comm(s)      | Compute(s) | Comm(s)      | Compute(s) | Comm(s)")
        print("-" * 100)
        
        # 2-GPU row
        print("2-GPU    |", end="")
        for batch_size in [32, 128, 512]:
            if batch_size in compute_times:
                compute = q3_results["compute_times"].get((batch_size, 2), 0)
                comm = q3_results["comm_times"].get((batch_size, 2), 0)
                print(f" {compute:10.4f} | {comm:12.4f} |", end="")
            else:
                print(" --------- | ----------- |", end="")
        print()
        
        # 4-GPU row
        print("4-GPU    |", end="")
        for batch_size in [32, 128, 512]:
            if batch_size in compute_times:
                compute = q3_results["compute_times"].get((batch_size, 4), 0)
                comm = q3_results["comm_times"].get((batch_size, 4), 0)
                print(f" {compute:10.4f} | {comm:12.4f} |", end="")
            else:
                print(" --------- | ----------- |", end="")
        print()
        
        # Q3.2: Communication bandwidth utilization
        print("\nQ3.2: Communication bandwidth utilization (ESTIMATED)")
        print("-" * 80)
        
        # Calculate bandwidth utilization
        def calculate_bandwidth(batch_size, gpu_count):
            comm_time = q3_results["comm_times"].get((batch_size, gpu_count), 0.001)  # Avoid div by zero
            data_volume = 2 * (gpu_count - 1) / gpu_count * model_size_bytes
            data_volume_gb = data_volume / (1024**3)
            return data_volume_gb / comm_time
        
        # Create Table 3
        print("\nTable 3: Communication Bandwidth utilization (ESTIMATED)")
        print("=" * 100)
        print("         | Batch-size 32 per GPU     | Batch-size 128 per GPU    | Batch-size 512 per GPU")
        print("         | Bandwidth (GB/s)          | Bandwidth (GB/s)          | Bandwidth (GB/s)")
        print("-" * 100)
        
        # 2-GPU row
        print("2-GPU    |", end="")
        for batch_size in [32, 128, 512]:
            if batch_size in compute_times:
                bandwidth = calculate_bandwidth(batch_size, 2)
                print(f" {bandwidth:24.2f} |", end="")
            else:
                print(" ---------------------- |", end="")
        print()
        
        # 4-GPU row
        print("4-GPU    |", end="")
        for batch_size in [32, 128, 512]:
            if batch_size in compute_times:
                bandwidth = calculate_bandwidth(batch_size, 4)
                print(f" {bandwidth:24.2f} |", end="")
            else:
                print(" ---------------------- |", end="")
        print()
        
        return q3_results
    
    # If we have valid Q2 results and multiple GPUs, calculate real values
    
    # Extract computation and communication times from Q2 results
    q3_results = {"compute_times": {}, "comm_times": {}}
    
    for batch_size in q2_results:
        # Single GPU has no communication overhead
        time_1gpu = q2_results[batch_size][1]
        q3_results["compute_times"][(batch_size, 1)] = time_1gpu
        q3_results["comm_times"][(batch_size, 1)] = 0
        
        # For multi-GPU, extract computation and communication components
        for gpu_count in q2_results[batch_size]:
            if gpu_count > 1:
                time_multi = q2_results[batch_size][gpu_count]
                
                # Estimate computation time as the same as single GPU
                # This is a simplification - in reality computation time might be slightly different
                compute_time = time_1gpu
                
                # Communication time is the difference
                comm_time = time_multi - compute_time if time_multi > compute_time else 0
                
                q3_results["compute_times"][(batch_size, gpu_count)] = compute_time
                q3_results["comm_times"][(batch_size, gpu_count)] = comm_time
    
    # Create Table 2
    print("\nTable 2: Compute and Communication time for different Batch Size")
    print("=" * 100)
    print("         | Batch-size 32 per GPU     | Batch-size 128 per GPU    | Batch-size 512 per GPU")
    print("         | Compute(s) | Comm(s)      | Compute(s) | Comm(s)      | Compute(s) | Comm(s)")
    print("-" * 100)
    
    # 2-GPU row
    if 2 in [gpu for _, gpu in q3_results["compute_times"]]:
        print("2-GPU    |", end="")
        for batch_size in [32, 128, 512]:
            if (batch_size, 2) in q3_results["compute_times"]:
                compute = q3_results["compute_times"][(batch_size, 2)]
                comm = q3_results["comm_times"][(batch_size, 2)]
                print(f" {compute:10.4f} | {comm:12.4f} |", end="")
            else:
                print(" --------- | ----------- |", end="")
        print()
    
    # 4-GPU row
    if 4 in [gpu for _, gpu in q3_results["compute_times"]]:
        print("4-GPU    |", end="")
        for batch_size in [32, 128, 512]:
            if (batch_size, 4) in q3_results["compute_times"]:
                compute = q3_results["compute_times"][(batch_size, 4)]
                comm = q3_results["comm_times"][(batch_size, 4)]
                print(f" {compute:10.4f} | {comm:12.4f} |", end="")
            else:
                print(" --------- | ----------- |", end="")
        print()
    
    # Visualize computation vs communication breakdown
    plt.figure(figsize=(15, 6))
    
    # Only include batch sizes up to 512 for visualization clarity
    plot_batch_sizes = [bs for bs in q2_results.keys() if bs <= 512]
    plot_batch_sizes.sort()
    
    # Plot for each GPU count
    gpu_counts = sorted([gpu for _, gpu in q3_results["compute_times"] if gpu > 1])
    
    for i, gpu_count in enumerate(gpu_counts, 1):
        plt.subplot(1, len(gpu_counts), i)
        
        x = np.arange(len(plot_batch_sizes))
        width = 0.4
        
        compute_data = [q3_results["compute_times"].get((bs, gpu_count), 0) for bs in plot_batch_sizes]
        comm_data = [q3_results["comm_times"].get((bs, gpu_count), 0) for bs in plot_batch_sizes]
        
        plt.bar(x, compute_data, width, label='Computation')
        plt.bar(x, comm_data, width, bottom=compute_data, label='Communication')
        
        plt.xlabel('Batch Size per GPU')
        plt.ylabel('Time (seconds)')
        plt.title(f'{gpu_count}-GPU Time Breakdown')
        plt.xticks(x, plot_batch_sizes)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('q3_time_breakdown.png')
    plt.show()
    
    # Q3.2: Communication bandwidth utilization
    print("\nQ3.2: Communication bandwidth utilization")
    print("-" * 80)
    
    # Calculate bandwidth utilization
    def calculate_bandwidth(batch_size, gpu_count):
        comm_time = q3_results["comm_times"].get((batch_size, gpu_count), 0.001)  # Avoid div by zero
        data_volume = 2 * (gpu_count - 1) / gpu_count * model_size_bytes
        data_volume_gb = data_volume / (1024**3)
        return data_volume_gb / comm_time if comm_time > 0 else 0
    
    bandwidth_results = {}
    for batch_size in q2_results:
        bandwidth_results[batch_size] = {}
        for gpu_count in q2_results[batch_size]:
            if gpu_count > 1:
                bandwidth_results[batch_size][gpu_count] = calculate_bandwidth(batch_size, gpu_count)
    
    # Create Table 3
    print("\nTable 3: Communication Bandwidth utilization")
    print("=" * 100)
    print("         | Batch-size 32 per GPU     | Batch-size 128 per GPU    | Batch-size 512 per GPU")
    print("         | Bandwidth (GB/s)          | Bandwidth (GB/s)          | Bandwidth (GB/s)")
    print("-" * 100)
    
    # 2-GPU row
    if 2 in gpu_counts:
        print("2-GPU    |", end="")
        for batch_size in [32, 128, 512]:
            if batch_size in bandwidth_results and 2 in bandwidth_results[batch_size]:
                bandwidth = bandwidth_results[batch_size][2]
                print(f" {bandwidth:24.2f} |", end="")
            else:
                print(" ---------------------- |", end="")
        print()
    
    # 4-GPU row
    if 4 in gpu_counts:
        print("4-GPU    |", end="")
        for batch_size in [32, 128, 512]:
            if batch_size in bandwidth_results and 4 in bandwidth_results[batch_size]:
                bandwidth = bandwidth_results[batch_size][4]
                print(f" {bandwidth:24.2f} |", end="")
            else:
                print(" ---------------------- |", end="")
        print()
    
    # Visualize bandwidth utilization
    if gpu_counts:
        plt.figure(figsize=(10, 6))
        
        for i, gpu_count in enumerate(gpu_counts):
            bandwidths = []
            x_values = []
            labels = []
            for batch_size in plot_batch_sizes:
                if batch_size in bandwidth_results and gpu_count in bandwidth_results[batch_size]:
                    x_values.append(batch_size)
                    bandwidths.append(bandwidth_results[batch_size][gpu_count])
                    labels.append(f"BS={batch_size}")
            
            if bandwidths:
                plt.plot(x_values, bandwidths, 'o-', label=f'{gpu_count} GPUs')
        
        plt.xlabel('Batch Size per GPU')
        plt.ylabel('Bandwidth Utilization (GB/s)')
        plt.title('Q3.2: Communication Bandwidth Utilization')
        plt.grid(True)
        plt.legend()
        plt.savefig('q3_bandwidth_utilization.png')
        plt.show()
    

    return q3_results

#############################################
# Run all experiments in sequence
#############################################

def main():
    # Load the dataset once to be used in all experiments
    print("Loading CIFAR-10 dataset...")
    trainset = load_cifar10()
    
    # Note the start time
    start_time = time.time()
    
    # Run Q1: Computational Efficiency w.r.t Batch Size
    q1_results = run_q1_batch_size_experiment(trainset)
    
    # Run Q2: Speedup Measurement (using results from Q1)
    q2_results = run_q2_speedup_experiment(trainset, q1_results)
    
    # Run Q3: Computation vs Communication (using results from Q1 and Q2)
    q3_results = run_q3_compute_vs_communication(trainset, q1_results, q2_results)
    
    # Total time taken
    total_time = time.time() - start_time
    print(f"\nTotal time for all experiments: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Final summary with key findings
    print("\n" + "="*80)
    print(" Summary of Key Findings ".center(80, '='))
    print("="*80)
    
    print("\nQ1: Computational Efficiency w.r.t Batch Size")
    if q1_results:
        min_batch, min_time = min(q1_results, key=lambda x: x[1])
        max_batch = max(q1_results, key=lambda x: x[0])[0]
        print(f"- Most efficient batch size: {min_batch} (Time: {min_time:.4f}s)")
        print(f"- Largest batch size that fits in GPU memory: {max_batch}")
    else:
        print("- No valid results")
    
    print("\nQ2: Speedup Measurement")
    if num_gpus >= 2 and q2_results:
        for batch_size in q2_results:
            if 2 in q2_results[batch_size]:
                speedup_2gpu = q2_results[batch_size][1] / q2_results[batch_size][2]
                print(f"- Batch size {batch_size}: 2-GPU speedup is {speedup_2gpu:.2f}x")
            if 4 in q2_results[batch_size]:
                speedup_4gpu = q2_results[batch_size][1] / q2_results[batch_size][4]
                print(f"- Batch size {batch_size}: 4-GPU speedup is {speedup_4gpu:.2f}x")
    else:
        print("- Not enough GPUs available to measure speedup")
    
    print("\nQ3: Computation vs Communication")
    if num_gpus >= 2 and q3_results:
        for key in sorted(q3_results["compute_times"].keys()):
            batch_size, gpu_count = key
            if gpu_count > 1:
                compute = q3_results["compute_times"][key]
                comm = q3_results["comm_times"][key]
                total = compute + comm
                compute_pct = compute / total * 100 if total > 0 else 0
                comm_pct = comm / total * 100 if total > 0 else 0
                
                print(f"- Batch size {batch_size}, {gpu_count} GPUs: " + 
                      f"Computation {compute_pct:.1f}%, Communication {comm_pct:.1f}%")
    else:
        print("- Not enough GPUs available to measure computation vs communication ratio")
    
    print("\nThank you for using this notebook! The experiment results can be used for your report.")
    
    return {
        "q1_results": q1_results,
        "q2_results": q2_results,
        "q3_results": q3_results
    }

# Run all experiments
if __name__ == "__main__":
    results = main()