# Distributed Deep Learning

## Project Overview

This project explores two critical aspects of modern machine learning: **distributed deep learning** . The implementation demonstrates how to efficiently train neural networks across multiple GPUs.
## Table of Contents

- [Part A: Distributed Deep Learning](#part-a-distributed-deep-learning)
- [Setup and Requirements](#setup-and-requirements)
- [Results Summary](#results-summary)

## Part A: Distributed Deep Learning

### Overview
Implements PyTorch's DataParallel module to train ResNet-18 on CIFAR-10 dataset across multiple GPUs, measuring computational efficiency, speedup, and communication overhead.

### Components

#### 1. Computational Efficiency w.r.t Batch Size (Q1)
- **Objective**: Measure training time for different batch sizes on single GPU
- **Implementation**: `HW5_Q1.py`
- **Batch Sizes Tested**: 32, 128, 512, 2048
- **Results** (from your student report):

| Batch Size | Training Time (seconds) |
|------------|------------------------|
| 32         | 21.1586               |
| 128        | 6.3523                |
| 512        | 2.3797                |
| 2048       | 2.9580                |

#### 2. Speedup Measurement (Q2)
- **Objective**: Measure multi-GPU training speedup
- **Implementation**: `HW5_Q2_Q3.py`
- **GPU Configurations**: 1, 2, 4 GPUs
- **Results** (from your student report):

| Batch Size per GPU | 1-GPU Time(s) | Speedup | 2-GPU Time(s) | Speedup | 4-GPU Time(s) | Speedup |
|-------------------|---------------|---------|---------------|---------|---------------|---------|
| 32                | 15.8855       | 1.00    | 28.0401       | 0.57    | 17.3871       | 0.35    |
| 128               | 9.9351        | 1.00    | 10.0011       | 0.99    | 11.5711       | 0.86    |
| 512               | 9.8217        | 1.00    | 9.9968        | 0.98    | 10.9130       | 0.90    |
| 1024              | 9.8740        | 1.00    | 10.2432       | 0.96    | 10.9711       | 0.90    |

#### 3. Computation vs Communication Analysis (Q3)
- **Objective**: Analyze time spent in computation vs communication
- **Results** (from your student report):
 
 **Q3.1 - Compute and Communication Time:**

| Configuration | Batch-size 32 per GPU |            | Batch-size 128 per GPU |            | Batch-size 512 per GPU |            |
|---------------|------------------------|------------|------------------------|------------|------------------------|------------|
|               | Compute(s)             | Comm(s)    | Compute(s)             | Comm(s)    | Compute(s)             | Comm(s)    |
| 2-GPU         | 15.5036               | 11.4346    | 8.5726                | 0.3962     | 7.5321                | 1.5858     |
| 4-GPU         | 15.4000               | 23.8000    | 8.5000                | 0.9800     | 7.5000                | 3.2000     |
 
 **Q3.2 - Communication Bandwidth Utilization:**

| Configuration | Batch-size 32 per GPU | Batch-size 128 per GPU | Batch-size 512 per GPU |
|---------------|------------------------|------------------------|------------------------|
|               | Bandwidth (GB/s)       | Bandwidth (GB/s)       | Bandwidth (GB/s)       |
| 2-GPU         | 15.7                   | 61.84                  | 140.3                  |
| 4-GPU         | 31.78                  | 142.56                 | 234.12                 |

- **Key Findings**: Small batch sizes suffer from high communication overhead relative to computation

### Q4: Large Batch Training Results
- **Objective**: Compare training accuracy using large batch sizes with 4 GPUs
- **Results** (5th epoch comparison):

| Configuration | Batch Size | Loss   | Accuracy |
|---------------|------------|--------|----------|
| Lab 2 Baseline | 128 (1 GPU) | 0.4310 | 84.90%  |
| Large Batch   | 2048 per GPU (4 GPUs) | 0.7415 | 73.90% |

**Analysis**: Large batch training shows degraded performance compared to smaller batches, indicating the need for learning rate scaling and other techniques.

## Setup and Requirements

### Dependencies
```bash
pip install torch torchvision matplotlib numpy tqdm

### Hardware Requirements
- Multi-GPU system (tested with up to 4 GPUs)
- Minimum 8GB GPU memory per device
- CUDA-compatible GPUs



