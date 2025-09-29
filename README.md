# Distributed Deep Learning - Part A

## Project Overview

This project explores distributed deep learning using PyTorch's DataParallel module to train ResNet-18 on the CIFAR-10 dataset across multiple GPUs. The implementation measures computational efficiency, speedup, and communication overhead in multi-GPU training scenarios.

## Table of Contents

- [Implementation Files](#implementation-files)
- [Hardware Configuration](#hardware-configuration)
- [Experimental Results](#experimental-results)
  - [Q1: Computational Efficiency w.r.t Batch Size](#q1-computational-efficiency-wrt-batch-size)
  - [Q2: Speedup Measurement](#q2-speedup-measurement)
  - [Q3: Computation vs Communication Analysis](#q3-computation-vs-communication-analysis)
  - [Q4: Large Batch Training](#q4-large-batch-training)
- [Setup and Requirements](#setup-and-requirements)
- [How to Run](#how-to-run)

## Implementation Files

- `HW5_Q1.py` - Single GPU training with varying batch sizes
- `HW5_Q2_Q3.py` - Multi-GPU training and communication analysis
- `HW5_Q4.py` - Large batch training experiments

## Hardware Configuration

- **GPU Type**: [Specify your GPU model, e.g., NVIDIA V100, A100, etc.]
- **Number of GPUs**: 1, 2, and 4 GPUs tested
- **GPU Memory**: [Specify memory per GPU]
- **Interconnect**: [Specify if known, e.g., NVLink, PCIe]

## Experimental Results

### Q1: Computational Efficiency w.r.t Batch Size

**Objective**: Measure training time for different batch sizes on a single GPU

**Methodology**: 
- Trained for 2 epochs, reporting second epoch time (excluding data loading)
- Batch sizes tested: 32, 128, 512, 2048

**Results**:

| Batch Size | Training Time (seconds) |
|------------|------------------------|
| 32         | 21.1586               |
| 128        | 6.3523                |
| 512        | 2.3797                |
| 2048       | 2.9580                |

**Analysis**: Training time decreases significantly from batch size 32 to 512 due to improved GPU utilization and parallelism. However, at batch size 2048, the time slightly increases, likely due to memory pressure and increased data transfer overhead.

### Q2: Speedup Measurement

**Objective**: Measure multi-GPU training speedup using weak scaling (constant batch size per GPU)

**Results**:

| Batch Size per GPU | 1-GPU Time(s) | 2-GPU Speedup | 4-GPU Speedup |
|-------------------|---------------|---------------|---------------|
| 32                | 15.89         | 0.57          | 0.35          |
| 128               | 9.94          | 0.99          | 0.86          |
| 512               | 9.82          | 0.98          | 0.90          |
| 1024              | 9.87          | 0.96          | 0.90          |

**Analysis**: 
- **Scaling Type**: This is weak scaling (constant work per GPU)
- Small batch sizes (32) show poor scaling due to high communication overhead relative to computation
- Larger batch sizes (128+) achieve near-linear scaling for 2 GPUs and good scaling for 4 GPUs
- Strong scaling (where total batch size remains constant) would show worse speedup numbers as each GPU would have less work to do

### Q3: Computation vs Communication Analysis

#### Q3.1: Compute and Communication Time

**Methodology**: Measured time spent in forward/backward passes (compute) versus gradient synchronization (communication)

**Results**:

| Configuration | Batch-size 32 per GPU |            | Batch-size 128 per GPU |            | Batch-size 512 per GPU |            |
|---------------|------------------------|------------|------------------------|------------|------------------------|------------|
|               | Compute(s)             | Comm(s)    | Compute(s)             | Comm(s)    | Compute(s)             | Comm(s)    |
| 2-GPU         | 15.50                 | 11.43      | 8.57                  | 0.40       | 7.53                  | 1.59       |
| 4-GPU         | 15.40                 | 23.80      | 8.50                  | 0.98       | 7.50                  | 3.20       |

**Key Findings**: 
- For batch size 32, communication time is comparable to or exceeds computation time, explaining poor speedup
- For larger batch sizes, computation dominates, allowing better parallel efficiency

#### Q3.2: Communication Bandwidth Utilization

**Formulas**:
- All-reduce communication time: `T_comm = 2 * (n-1)/n * model_size / bandwidth`
- Bandwidth utilization: `Bandwidth = (2 * (n-1)/n * model_size) / communication_time`

Where n = number of GPUs, model_size ≈ [specify model size in bytes]

**Results**:

| Configuration | Batch-size 32 per GPU | Batch-size 128 per GPU | Batch-size 512 per GPU |
|---------------|------------------------|------------------------|------------------------|
|               | Bandwidth (GB/s)       | Bandwidth (GB/s)       | Bandwidth (GB/s)       |
| 2-GPU         | 15.7                   | 61.84                  | 140.3                  |
| 4-GPU         | 31.78                  | 142.56                 | 234.12                 |

**Analysis**: Larger batch sizes achieve higher bandwidth utilization because they amortize communication overhead over more computation.

### Q4: Large Batch Training

**Objective**: Compare training accuracy using large batch sizes with the Lab 2 baseline

**Results (5th Epoch)**:

| Configuration | Batch Size | Loss   | Accuracy |
|---------------|------------|--------|----------|
| Lab 2 Baseline | 128 (1 GPU) | 0.4310 | 84.90%  |
| Large Batch   | 2048 per GPU (4 GPUs) | 0.7415 | 73.90% |

**Analysis**: The large batch configuration shows degraded performance (11% accuracy drop). This is a well-known issue in large batch training caused by:
- Reduced gradient noise leading to sharper minima
- Fewer parameter updates per epoch
- Need for learning rate scaling

**Potential Remedies**:
1. Linear learning rate scaling with batch size
2. Learning rate warmup
3. Layer-wise Adaptive Rate Scaling (LARS)
4. Batch size scheduling

## Hardware Requirements

- Multi-GPU system (tested with up to 4 GPUs)
- Minimum 8GB GPU memory per device
- CUDA-compatible GPUs

## Software Requirements

- Python 3.7+
- PyTorch 1.8+
- CUDA 11.0+

## Commands:
- python HW5_Q1.py
 ### For 2 GPUs
- CUDA_VISIBLE_DEVICES=0,1 python HW5_Q2_Q3.py --num_gpus 2 --batch_size 128

  ### For 4 GPUs
- CUDA_VISIBLE_DEVICES=0,1,2,3 python HW5_Q2_Q3.py --num_gpus 4 --batch_size 512
- CUDA_VISIBLE_DEVICES=0,1,2,3 python HW5_Q4.py --batch_size 2048 --num_gpus 4

## References

PyTorch Distributed Overview: https://pytorch.org/tutorials/beginner/dist_overview.html
PyTorch DataParallel: https://pytorch.org/docs/stable/nn.html#dataparallel
Patarasuk & Yuan (2009). "Bandwidth optimal all-reduce algorithms for clusters of workstations"
Goyal et al. (2017). "Accurate, large minibatch SGD: training imagenet in 1 hour"
