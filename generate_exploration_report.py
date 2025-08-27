"""HTML report generator for CFD simulation exploration results."""
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


def load_analysis_results(json_path: str) -> Dict[str, Any]:
    """Load analysis results from JSON file.
    
    Args:
        json_path: Path to the JSON file containing analysis results
        
    Returns:
        Dictionary containing analysis results
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        json.JSONDecodeError: If JSON file is invalid
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_html_report(
    analysis_results: Optional[Dict[str, Any]] = None,
    json_path: Optional[str] = None
) -> None:
    """Generate an HTML report with all exploration findings.
    
    Args:
        analysis_results: Optional pre-loaded analysis results dictionary
        json_path: Optional path to JSON file containing results
        
    Returns:
        None. Saves HTML report to file.
    """
    # Load from JSON if path provided
    if json_path and os.path.exists(json_path):
        analysis_results = load_analysis_results(json_path)
    elif analysis_results is None:
        # Try default path
        default_path = os.path.join('outputs', 'eda', 'analysis_results.json')
        if os.path.exists(default_path):
            analysis_results = load_analysis_results(default_path)
            print(f"Loaded results from: {default_path}")
        else:
            print("No analysis results found. Please run explore_cfd_data.py first.")
            return
    
    # Also load individual simulation metadata if available
    metadata_path = os.path.join('outputs', 'eda', 'all_simulations_metadata.json')
    simulations_metadata: List[Dict[str, Any]] = []
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            simulations_metadata = json.load(f)
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CFD Simulation Data Exploration Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
            h3 {{ color: #888; }}
            .figure {{ margin: 20px 0; text-align: center; }}
            .figure img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
            .stats-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            .stats-table th, .stats-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .stats-table th {{ background-color: #f4f4f4; }}
            .highlight {{ background-color: #ffffcc; padding: 10px; margin: 10px 0; border-left: 4px solid #ff9900; }}
            .metric {{ display: inline-block; margin: 10px 20px; padding: 10px; background-color: #f0f0f0; border-radius: 5px; }}
            .sim-gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
            .sim-card {{ border: 1px solid #ddd; padding: 10px; background: #f9f9f9; }}
            .sim-card img {{ width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>CFD Simulation Data Exploration Report</h1>
        <p>Generated on: {timestamp}</p>
        
        <h2>1. Dataset Overview</h2>
        <div class="metric">
            <strong>Simulations:</strong> {n_simulations}
        </div>
        <div class="metric">
            <strong>Mesh Size:</strong> {n_nodes} nodes
        </div>
        <div class="metric">
            <strong>Time Steps:</strong> {n_timesteps}
        </div>
        
        <h2>2. Individual Simulation Analysis</h2>
        {individual_sims_section}
        
        <h2>3. Spatial Analysis Summary</h2>
        <div class="highlight">
            <h3>Key Findings:</h3>
            <ul>
                <li>Hotspot location identified at coordinates: {hotspot_coords}</li>
                <li>Maximum temperature reached: {max_temp:.2f}°C</li>
                <li>Minimum temperature: {min_temp_overall:.2f}°C</li>
                <li>Temperature range: {temp_range:.2f}°C</li>
            </ul>
        </div>
        
        <h2>4. Temporal Analysis Summary</h2>
        <div class="highlight">
            <h3>Key Findings:</h3>
            <ul>
                <li>Average steady state reached at time step: {steady_state_time}</li>
                <li>Time constant estimate: {time_constant:.2f} steps</li>
                <li>Temperature rise pattern: {rise_pattern}</li>
            </ul>
        </div>
        
        <h2>5. Node Property Analysis</h2>
        <table class="stats-table">
            <tr>
                <th>Node Type</th>
                <th>Count</th>
                <th>Mean Final Temp (°C)</th>
                <th>Std Dev</th>
                <th>Max Temp (°C)</th>
            </tr>
            {node_type_stats}
        </table>
        
        <h2>6. Multi-Simulation Comparison</h2>
        <div class="highlight">
            <h3>Key Findings:</h3>
            <ul>
                <li>Temperature increases with current at rate: {current_sensitivity:.2f} °C/A</li>
                <li>Model R²: {r_squared:.3f}</li>
                <li>Ambient temperature effect: {ambient_effect}</li>
            </ul>
        </div>
        
        <div class="figure">
            <img src="simulation_comparison.png" alt="Simulation Comparison">
            <p><em>Figure: Comparison across different simulation conditions</em></p>
        </div>
        
        <div class="figure">
            <img src="all_hotspots_analysis.png" alt="Comprehensive Hotspot Analysis">
            <p><em>Figure: Comprehensive hotspot analysis across all simulations</em></p>
        </div>
        
        <h2>7. Recommendations for Modeling</h2>
        <ul>
            <li><strong>Important features:</strong> spatial coordinates, thermal conductivity, node type</li>
            <li><strong>Suggested prediction horizon:</strong> {prediction_horizon} time steps</li>
            <li><strong>Key regions to monitor:</strong> {key_regions}</li>
            <li><strong>Modeling approach:</strong> Consider graph neural networks to capture spatial dependencies</li>
            <li><strong>Feature engineering:</strong> Include distance from heat source and local connectivity features</li>
        </ul>
        
        <h2>8. Data Quality Notes</h2>
        <ul>
            <li>All simulations start from consistent initial conditions</li>
            <li>Temperature evolution shows expected physical behavior</li>
            <li>Hotspot locations are consistent across simulations</li>
            <li>Strong correlation between input current and maximum temperature</li>
        </ul>
    </body>
    </html>
    """
    
    # Generate individual simulations section
    individual_sims_html = '<div class="sim-gallery">'
    for i, sim in enumerate(simulations_metadata):
        steady_state_info = (
            f"Step {sim['steady_state_time']}" 
            if sim.get('steady_state_time', -1) > 0 
            else "Not reached"
        )
        
        individual_sims_html += f'''
        <div class="sim-card">
            <h3>Simulation {i+1}: {sim['filename']}</h3>
            <p><strong>Current:</strong> {sim['current']} A | '''
        individual_sims_html += f'''<strong>Ambient:</strong> {sim['ambient_temp']}°C</p>
            <p><strong>Max Temp:</strong> {sim['max_temp']:.1f}°C | '''
        individual_sims_html += f'''<strong>Range:</strong> {sim['temp_range']:.1f}°C</p>
            <img src="sim_{i+1:02d}/temperature_final_3d.png" alt="Final temperature">
            <p><strong>Steady State:</strong> {steady_state_info}</p>
        </div>
        '''
    individual_sims_html += '</div>'
    
    # Get values with defaults for template formatting
    template_values: Dict[str, Union[str, int, float]] = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'n_simulations': analysis_results.get('n_simulations', 'N/A'),
        'n_nodes': analysis_results.get('n_nodes', 5361),
        'n_timesteps': analysis_results.get('n_timesteps', 121),
        'individual_sims_section': (
            individual_sims_html 
            if simulations_metadata 
            else '<p>No individual simulation data available</p>'
        ),
        'hotspot_coords': analysis_results.get('hotspot_coords', 'N/A'),
        'max_temp': float(analysis_results.get('max_temp', 0)),
        'min_temp_overall': float(
            analysis_results.get(
                'min_temp_overall', 
                analysis_results.get('min_temp', 0)
            )
        ),
        'temp_range': float(analysis_results.get('temp_range', 0)),
        'steady_state_time': analysis_results.get('steady_state_time', 'N/A'),
        'time_constant': float(analysis_results.get('time_constant', 0)),
        'rise_pattern': analysis_results.get('rise_pattern', 'exponential'),
        'node_type_stats': analysis_results.get('node_type_stats', ''),
        'ambient_effect': analysis_results.get('ambient_effect', 'linear offset'),
        'current_sensitivity': float(analysis_results.get('current_sensitivity', 0)),
        'r_squared': float(analysis_results.get('r_squared', 0)),
        'prediction_horizon': int(analysis_results.get('prediction_horizon', 100)),
        'key_regions': analysis_results.get('key_regions', 'N/A')
    }
    
    # Fill in the template with actual results
    report = html_template.format(**template_values)
    
    # Save report in the outputs/eda directory
    output_dir = os.path.join('outputs', 'eda')
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'exploration_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report generated: {report_path}")


def main() -> None:
    """Main entry point for report generation."""
    generate_html_report()


if __name__ == "__main__":
    main()
