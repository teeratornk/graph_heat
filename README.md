# Graph Heat Project

A machine learning project for analyzing and predicting heat distribution using graph neural networks on CFD (Computational Fluid Dynamics) data.

## Project Overview

This project implements a complete pipeline for processing CFD data, converting it to graph representations, and training graph neural networks for heat prediction tasks.

## Example Outputs

### Exploratory Data Analysis Results

The EDA pipeline generates comprehensive visualizations and reports. Here are some example outputs:

#### 1. Interactive HTML Report
<div align="center">
  <img src="assets/exploration_report.html" alt="Exploration Report">
  <p><em>Interactive HTML report with comprehensive analysis findings</em></p>
</div>

The HTML report (`exploration_report.html`) provides:
- Complete dataset overview with interactive elements
- Individual simulation analysis cards
- Statistical summaries and correlations
- Modeling recommendations based on data characteristics
- [View Report](assets/exploration_report.html)

#### 2. Temperature Distribution with Node Types
<div align="center">
  <img src="assets/temperature_final_3d_with_node_types.png" alt="3D Temperature with Node Types" width="80%">
  <p><em>3D visualization showing temperature distribution with different node types (interior, boundary, interface)</em></p>
</div>

This visualization shows:
- **Interior nodes** (circles): Main mesh points with standard thermal properties
- **Boundary nodes** (triangles): Edge nodes with boundary conditions
- **Interface nodes** (squares): Special nodes at material interfaces
- Color mapping represents temperature magnitude (hot colors = higher temperatures)

#### 3. Multi-Simulation Comparison
<div align="center">
  <img src="assets/simulation_comparison.png" alt="Simulation Comparison" width="90%">
  <p><em>Comparison of temperature evolution across different simulation conditions</em></p>
</div>

Key insights from this comparison:
- Left plot: Temperature evolution over time for different current/ambient conditions
- Right plot: Correlation between input current and maximum temperature
- Color coding indicates ambient temperature effects
- Clear linear relationship with R² > 0.9

#### 4. Comprehensive Hotspot Analysis
<div align="center">
  <img src="assets/all_hotspots_analysis.png" alt="Hotspot Analysis" width="90%">
  <p><em>Statistical analysis of hotspot locations and temperatures across all simulations</em></p>
</div>

This comprehensive analysis reveals:
- **Top-left**: Current vs hotspot temperature relationship
- **Top-right**: Spatial distribution of hotspot locations
- **Bottom-left**: Temperature histogram showing hotspot distribution
- **Bottom-right**: Most frequent hotspot nodes for targeted monitoring

## Project Workflow

### Step 1: Exploratory Data Analysis (EDA)
**Main Script:** `generate_exploration_report.py`

This is the entry point for data exploration that orchestrates the entire EDA process:

#### Functionality:
- **Automated Pipeline**: Runs `explore_cfd_data.py` as a subprocess to perform comprehensive data analysis
- **Report Generation**: Creates an interactive HTML report with all exploration findings
- **Results Aggregation**: Combines analysis results from multiple sources into a unified report

#### Process Flow:
1. Executes `explore_cfd_data.py` via subprocess to analyze CFD simulation data
2. Loads analysis results from JSON files:
   - `outputs/eda/analysis_results.json` - Main analysis metrics
   - `outputs/eda/all_simulations_metadata.json` - Individual simulation details
3. Generates comprehensive HTML report with:
   - Dataset overview (number of simulations, mesh size, time steps)
   - Individual simulation analysis cards with visualizations
   - Spatial analysis summary (hotspot locations, temperature distributions)
   - Temporal analysis (steady-state times, temperature evolution patterns)
   - Node property statistics by type
   - Multi-simulation comparison and correlations
   - Modeling recommendations based on findings
   - Data quality assessment

#### Output Files:
- `outputs/eda/exploration_report.html` - Main HTML report
- `outputs/eda/analysis_results.json` - Structured analysis data
- `outputs/eda/all_simulations_metadata.json` - Per-simulation metadata
- `outputs/eda/sim_*/` - Individual simulation visualizations

#### Key Insights Generated:
- Temperature-current relationship sensitivity
- Hotspot identification and characterization
- Steady-state convergence analysis
- Feature importance for modeling
- Recommended prediction horizons

#### Usage:
```bash
# Run complete EDA pipeline and generate report
python generate_exploration_report.py

# The script will automatically:
# 1. Run explore_cfd_data.py
# 2. Process all simulation files
# 3. Generate visualizations
# 4. Create HTML report
```

#### Subprocess: `explore_cfd_data.py`

This script performs the detailed analysis of CFD simulation data:

##### Core Functions:

1. **Data Loading (`load_simulation_data`)**:
   - Loads HDF5 simulation files from `data/` directory
   - Extracts metadata from filenames (current and ambient temperature)
   - Handles both `.hdf5` and `.h5` file formats
   - Parses file patterns like `I=1500_T=15` for current and temperature values

2. **Data Structure Exploration (`explore_data_structure`)**:
   - Identifies and maps data arrays:
     - `node_pos`: 3D coordinates of mesh nodes (5361 nodes × 3 dimensions)
     - `temperature`: Temperature evolution (121 time steps × 5361 nodes)
     - `k`: Thermal conductivity values for each node
     - `node_types`: Node classification (interior, boundary, interface)
     - `edge_src/edge_dst`: Graph connectivity information
   - Handles 3D temperature arrays by squeezing unnecessary dimensions

3. **Spatial Analysis**:
   - **3D Temperature Visualization**: Creates 3D scatter plots with temperature color mapping
   - **Node Type Analysis**: Visualizes different node types with distinct markers:
     - Type 0: Interior nodes (circles)
     - Type 1: Boundary nodes (triangles)
     - Type 2: Interface/Special nodes (squares)
   - **2D Projections**: Generates XY, XZ, and YZ projections with node type overlays
   - **Hotspot Identification**: Finds nodes with highest temperatures and their coordinates

4. **Temporal Analysis**:
   - **Evolution Tracking**: Plots temperature vs time for selected nodes
   - **Steady-State Detection**: Determines convergence using rate-of-change threshold (0.01°C)
   - **Time Constant Estimation**: Calculates thermal response time (63.2% of final temperature)
   - **Pattern Recognition**: Identifies temperature rise patterns (exponential/linear)

5. **Node Property Analysis (`analyze_node_properties`)**:
   - Correlates temperature with:
     - Thermal conductivity (k values)
     - Node types (boundary conditions)
     - Spatial coordinates (X, Y, Z)
   - Generates correlation matrix and statistical summaries
   - Creates box plots for temperature distribution by node type

6. **Multi-Simulation Comparison (`compare_simulations`)**:
   - Compares temperature evolution across different operating conditions
   - Analyzes current-temperature relationships
   - Studies ambient temperature effects
   - Calculates sensitivity metrics (°C/A)
   - Performs linear regression with R² calculation

7. **Comprehensive Hotspot Analysis (`analyze_all_hotspots`)**:
   - Aggregates hotspot data across all simulations
   - Identifies most frequent hotspot locations
   - Analyzes hotspot temperature distributions
   - Creates frequency maps of critical regions

##### Visualization Outputs:
For each simulation, generates:
- `temperature_initial_3d.png`: Initial temperature distribution
- `temperature_final_3d.png`: Final steady-state temperature
- `temperature_final_3d_with_node_types.png`: Temperature with node type markers
- `temperature_by_node_type_separate.png`: Separate views for each node type
- `temperature_final_2d_*_with_node_types.png`: 2D projections (XY, XZ, YZ)
- `temperature_evolution.png`: Time series for representative nodes
- `node_property_analysis.png`: Property correlation analysis

Comparative visualizations:
- `simulation_comparison.png`: Cross-simulation temperature evolution
- `all_hotspots_analysis.png`: Comprehensive hotspot statistics

##### Data Files Generated:
- `outputs/eda/analysis_results.json`: Main analysis metrics and findings
- `outputs/eda/all_simulations_metadata.json`: Per-simulation metadata
- `outputs/eda/hotspots_summary.csv`: Detailed hotspot information
- `outputs/eda/sim_*/`: Individual simulation results folders

##### Key Metrics Extracted:
- **Spatial Metrics**:
  - Maximum/minimum temperatures and locations
  - Temperature gradients and distributions
  - Hotspot coordinates and intensities
  - Node type temperature statistics

- **Temporal Metrics**:
  - Steady-state convergence time
  - Time constants for thermal response
  - Temperature rise rates
  - Stability indicators

- **Correlation Metrics**:
  - Current sensitivity (°C/A)
  - Ambient temperature effects
  - R² values for predictive models
  - Feature importance rankings

##### Usage Notes:
- Automatically processes all HDF5 files in `data/` directory
- Handles varying mesh sizes and time steps
- Robust to missing data fields
- Creates comprehensive visualization suite
- Saves all results in JSON format for downstream processing

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

- Python 3.10+
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
