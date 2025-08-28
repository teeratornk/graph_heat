# Graph Heat Project

A machine learning project for analyzing and predicting heat distribution using graph neural networks on CFD (Computational Fluid Dynamics) data.

## Project Overview

This project implements a complete pipeline for processing CFD data, converting it to graph representations, and training graph neural networks for heat prediction tasks.

## Project Workflow

### Step 1: Exploratory Data Analysis (EDA)
**Script:** `generate_exploration_report.py`
- Generates comprehensive exploration reports for the CFD dataset
- Internally calls `explore_cfd_data.py` as a subprocess to perform detailed analysis
- Produces visualizations and statistics about the data distribution

### Step 2: HDF5 Data Analysis
**Script:** `batch_analyze_hdf5.py`
- Performs batch analysis on HDF5 formatted data files
- Calls `analyze_hdf5_metadata.py` to interrogate and extract metadata
- Provides insights into data structure and properties

### Step 3: Data Preprocessing
**Script:** `preprocess_gis_to_pyg.py`
- Converts GIS data to PyTorch Geometric (PyG) format
- Prepares graph representations from spatial data
- Internally runs `train_val_test_split.py` to generate:
  - Training set
  - Validation set
  - Test set

### Step 4: Data Normalization
**Script:** `normalizer.py`
- Normalizes the dataset based on training set statistics
- Ensures consistent scaling across all data splits
- Saves normalization parameters for inference

### Step 5: Model Training
**Core Components:**
- `model.py`: Contains the graph neural network architecture
- `sampler.py`: Implements sampling strategies for batch creation
- `train.py`: Main training script that:
  - Loads the model from `model.py`
  - Uses sampling strategies from `sampler.py`
  - Trains the model with specified hyperparameters

**Shell Scripts for Training:**
- `train_basic.sh`: Basic training configuration
- `train_full.sh`: Full training with all features enabled
- `train_distributed.sh`: Distributed training across multiple GPUs (if available)

### Step 6: Model Testing
**Script:** `test.py`
- Evaluates the best trained model on the test dataset
- Generates performance metrics and visualizations

**Shell Scripts for Testing:**
- `test_best.sh`: Tests the best checkpoint
- `test_all.sh`: Tests all saved checkpoints
- `test_visualize.sh`: Tests with visualization outputs

### Step 7: Diagnostics
**Diagnostic Scripts:**
- `diagnose_data.py`: Identifies and reports data quality issues
- `diagnose_test_issue.py`: Troubleshoots testing problems

**Shell Scripts for Diagnostics:**
- `diagnose_pipeline.sh`: Runs full pipeline diagnostics
- `diagnose_memory.sh`: Checks memory usage and optimization
- `diagnose_performance.sh`: Analyzes model performance bottlenecks

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd graph_heat

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Running the Complete Pipeline

```bash
# 1. Explore the data
python generate_exploration_report.py

# 2. Analyze HDF5 files
python batch_analyze_hdf5.py

# 3. Preprocess and split data
python preprocess_gis_to_pyg.py

# 4. Normalize data
python normalizer.py

# 5. Train the model
bash train_basic.sh  # or train_full.sh for full training

# 6. Test the model
bash test_best.sh

# 7. Run diagnostics if needed
bash diagnose_pipeline.sh
```

## Shell Script Usage

### Training Scripts

- **`train_basic.sh`**: Quick training with default parameters
  ```bash
  bash train_basic.sh
  ```

- **`train_full.sh`**: Complete training with optimized hyperparameters
  ```bash
  bash train_full.sh
  ```

- **`train_distributed.sh`**: Multi-GPU distributed training
  ```bash
  bash train_distributed.sh
  ```

### Testing Scripts

- **`test_best.sh`**: Evaluate the best model checkpoint
  ```bash
  bash test_best.sh
  ```

- **`test_all.sh`**: Evaluate all saved checkpoints
  ```bash
  bash test_all.sh
  ```

- **`test_visualize.sh`**: Generate visualizations during testing
  ```bash
  bash test_visualize.sh
  ```

### Diagnostic Scripts

- **`diagnose_pipeline.sh`**: Check entire pipeline health
  ```bash
  bash diagnose_pipeline.sh
  ```

- **`diagnose_memory.sh`**: Analyze memory usage
  ```bash
  bash diagnose_memory.sh
  ```

- **`diagnose_performance.sh`**: Profile performance metrics
  ```bash
  bash diagnose_performance.sh
  ```

## Directory Structure

```
graph_heat/
├── data/                   # Raw and processed data
│   ├── raw/               # Original CFD/GIS data
│   ├── processed/         # PyG formatted data
│   └── splits/            # Train/val/test splits
├── models/                 # Saved model checkpoints
├── logs/                   # Training and testing logs
├── reports/                # EDA and analysis reports
├── scripts/                # Shell scripts
└── src/                    # Python source code
```

## Configuration

Configuration parameters can be modified in:
- `config.yaml`: Main configuration file
- Individual shell scripts for specific run configurations

## Requirements

- Python 3.8+
- PyTorch 1.9+
- PyTorch Geometric
- NumPy
- Pandas
- H5py
- Additional dependencies in `requirements.txt`

## License

[Specify your license here]

## Contact

[Your contact information]
