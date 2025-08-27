# CFD Simulation Data Exploration

This project contains scripts for exploring and analyzing CFD simulation data with spatial and temporal patterns.

## Project Structure

```
graph_data/
├── data/                      # Directory containing simulation .npz files
├── exploration_results/       # Generated visualizations and reports
├── explore_cfd_data.py       # Main exploration script
├── generate_exploration_report.py  # HTML report generator
└── README.md                 # This file
```

## Scripts Overview

### 1. `explore_cfd_data.py`
Main exploration script that performs comprehensive analysis:
- **Spatial Analysis**: 3D visualization of temperature distribution on mesh
- **Temporal Analysis**: Temperature evolution over time
- **Node Property Analysis**: Correlation between thermal properties and temperature
- **Multi-simulation Comparison**: Analysis across different operating conditions

### 2. `generate_exploration_report.py`
Generates an HTML report summarizing all findings from the exploration.

## Data Format Expected

The scripts expect `.npz` files in the `data/` directory with the following structure:
- `node_coords`: (5361, 3) - 3D coordinates of mesh nodes
- `connectivity`: Mesh connectivity information
- `temperatures`: (120, 5361) - Temperature values over time
- `k_values`: (5361,) - Thermal conductivity values
- `node_types`: (5361,) - Node type indicators (0, 1, or 2)

File naming convention: `sim_{current}_{ambient_temp}.npz`

## Usage

1. Place all simulation `.npz` files in the `data/` directory
2. Run the exploration script from the project root:
   ```bash
   python explore_cfd_data.py
   ```
3. Generate HTML report:
   ```bash
   python generate_exploration_report.py
   ```

## Output Files

All outputs are saved in the `exploration_results/` directory:
- `temperature_initial_3d.png`: Initial temperature distribution
- `temperature_final_3d.png`: Final (steady-state) temperature distribution
- `temperature_evolution.png`: Time series of temperature at selected nodes
- `node_property_analysis.png`: Analysis of node properties vs temperature
- `simulation_comparison.png`: Comparison across different simulations
- `exploration_report.html`: Comprehensive HTML report

## Key Insights Expected

1. **Hotspot Identification**: Location and magnitude of maximum temperatures
2. **Steady-State Behavior**: Time to reach thermal equilibrium
3. **Material Properties**: Effect of thermal conductivity on temperature
4. **Operating Conditions**: How current and ambient temperature affect results
5. **Spatial Patterns**: Temperature gradients and heat flow paths

## Customization

Modify the following parameters in `explore_cfd_data.py`:
- `n_hotspots`: Number of hottest nodes to identify
- `threshold`: Steady-state detection threshold
- Node selection for temporal analysis
- Number of simulations to compare

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- Pandas
- Seaborn

Install dependencies:
```bash
pip install numpy matplotlib pandas seaborn
```
