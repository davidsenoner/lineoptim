"""
Simple LineOptim Example
========================

This example demonstrates the basic usage of LineOptim for electrical power line simulation.
"""

import torch
from collections import OrderedDict
import lineoptim as lo

def main():
    """Simple example demonstrating LineOptim basic usage."""
    
    print("LineOptim Simple Example")
    print("=" * 24)
    
    # Define voltage and core configuration
    v_nominal = torch.tensor([400.0, 400.0, 400.0])  # Three-phase nominal voltage
    cores = OrderedDict({"L1": "brown", "L2": "black", "L3": "grey"})
    
    # Create main power line
    main_line = lo.Line(
        name="Main power line",
        position=0,
        resistivity=torch.tensor([0.15, 0.15, 0.15]),  # Ohm/km per phase
        reactance=torch.tensor([0.0, 0.0, 0.0]),
        v_nominal=v_nominal,
        cores=cores
    )
    
    # Add loads to the line
    print("\nAdding loads to the line...")
    main_line.add("Load 1", 100, active_power=2000, v_nominal=v_nominal, power_factor=0.91)
    main_line.add("Load 2", 300, active_power=5000, v_nominal=v_nominal, power_factor=0.85)
    main_line.add("Load 3", 500, active_power=8000, v_nominal=v_nominal, power_factor=0.90)
    
    # Compute electrical parameters
    print("Computing electrical parameters...")
    main_line.recompute()
    
    # Display results
    print("\nResults:")
    print(f"Number of loads: {len(main_line)}")
    print(f"Voltage range: {main_line.voltage.min().min().item():.2f} - {main_line.voltage.max().max().item():.2f} V")
    print(f"Current range: {main_line.current.min().min().item():.2f} - {main_line.current.max().max().item():.2f} A")
    print(f"Voltage unbalance: {main_line.voltage_unbalance.mean().item():.2f}%")
    
    # Network optimization
    print("\nOptimizing network...")
    network = lo.Network()
    network.add(main_line)
    network.optimize(epochs=50, lr=0.01, max_v_drop=5.0)
    
    print("\nOptimization completed successfully!")
    print(f"New voltage range: {main_line.voltage.min().min().item():.2f} - {main_line.voltage.max().max().item():.2f} V")
    
    # Save configuration
    config_file = "simple_example_config.json"
    main_line.save_to_json(config_file)
    print(f"\nConfiguration saved to: {config_file}")

if __name__ == "__main__":
    main()