"""
Utility functions for LineOptim package.

This module provides convenience functions for common electrical calculations
and power line creation patterns.
"""

import torch
from collections import OrderedDict
from typing import List, Dict, Any, Union, Tuple, Optional

from lineoptim.components import Line


def create_three_phase_line(
    name: str,
    position: float = 0,
    voltage: float = 400.0,
    resistivity: Union[float, List[float]] = 0.15,
    reactance: Union[float, List[float]] = 0.0,
    core_colors: Optional[Dict[str, str]] = None
) -> Line:
    """
    Create a standard three-phase power line.
    
    Args:
        name: Name of the line
        position: Position where line is connected (meters)
        voltage: Nominal voltage per phase (V)
        resistivity: Conductor resistivity (ohm/km per phase)
        reactance: Conductor reactance (ohm/km per phase)
        core_colors: Optional dictionary of core colors
        
    Returns:
        Configured three-phase Line instance
    """
    # Default core colors for three-phase system
    if core_colors is None:
        core_colors = {"L1": "brown", "L2": "black", "L3": "grey"}
    
    cores = OrderedDict(core_colors)
    v_nominal = torch.tensor([voltage] * 3)
    
    # Handle resistivity and reactance
    if isinstance(resistivity, (int, float)):
        resistivity = torch.tensor([resistivity] * 3)
    else:
        resistivity = torch.tensor(resistivity)
        
    if isinstance(reactance, (int, float)):
        reactance = torch.tensor([reactance] * 3)
    else:
        reactance = torch.tensor(reactance)
    
    return Line(
        name=name,
        position=position,
        cores=cores,
        resistivity=resistivity,
        reactance=reactance,
        v_nominal=v_nominal
    )


def create_single_phase_line(
    name: str,
    position: float = 0,
    voltage: float = 230.0,
    resistivity: float = 0.15,
    reactance: float = 0.0,
    core_color: str = "blue"
) -> Line:
    """
    Create a single-phase power line.
    
    Args:
        name: Name of the line
        position: Position where line is connected (meters)
        voltage: Nominal voltage (V)
        resistivity: Conductor resistivity (ohm/km)
        reactance: Conductor reactance (ohm/km)
        core_color: Color for the core
        
    Returns:
        Configured single-phase Line instance
    """
    cores = OrderedDict({"L1": core_color})
    v_nominal = torch.tensor([voltage])
    resistivity = torch.tensor([resistivity])
    reactance = torch.tensor([reactance])
    
    return Line(
        name=name,
        position=position,
        cores=cores,
        resistivity=resistivity,
        reactance=reactance,
        v_nominal=v_nominal
    )


def add_residential_load(
    line: Line,
    name: str,
    position: float,
    power_kw: float,
    voltage: Optional[torch.Tensor] = None,
    power_factor: float = 0.9
) -> None:
    """
    Add a residential load to a power line.
    
    Args:
        line: Line instance to add load to
        name: Name of the load
        position: Position along the line (meters)
        power_kw: Active power in kW
        voltage: Nominal voltage (defaults to line voltage)
        power_factor: Power factor (0 < power_factor <= 1)
    """
    if voltage is None:
        voltage = line['v_nominal']
    
    active_power = power_kw * 1000  # Convert kW to W
    
    line.add(
        name=name,
        position=position,
        active_power=active_power,
        v_nominal=voltage,
        power_factor=power_factor
    )


def add_industrial_load(
    line: Line,
    name: str,
    position: float,
    power_kw: float,
    voltage: Optional[torch.Tensor] = None,
    power_factor: float = 0.85
) -> None:
    """
    Add an industrial load to a power line.
    
    Args:
        line: Line instance to add load to
        name: Name of the load
        position: Position along the line (meters)
        power_kw: Active power in kW
        voltage: Nominal voltage (defaults to line voltage)
        power_factor: Power factor (0 < power_factor <= 1)
    """
    if voltage is None:
        voltage = line['v_nominal']
    
    active_power = power_kw * 1000  # Convert kW to W
    
    line.add(
        name=name,
        position=position,
        active_power=active_power,
        v_nominal=voltage,
        power_factor=power_factor
    )


def calculate_voltage_drop_percentage(
    line: Line,
    reference_voltage: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Calculate voltage drop percentage along the line.
    
    Args:
        line: Line instance
        reference_voltage: Reference voltage (defaults to line nominal voltage)
        
    Returns:
        Voltage drop percentage tensor
    """
    if reference_voltage is None:
        reference_voltage = line['v_nominal']
    
    voltages = torch.stack([load['v_nominal'] for load in line['loads']])
    voltage_drop = (1 - voltages / reference_voltage) * 100
    
    return voltage_drop


def get_power_summary(line: Line) -> Dict[str, float]:
    """
    Get power summary for a line.
    
    Args:
        line: Line instance
        
    Returns:
        Dictionary with power statistics
    """
    active_powers = [load['active_power'] for load in line['loads']]
    apparent_powers = [load['apparent_power'] for load in line['loads']]
    
    return {
        'total_active_power_kw': sum(active_powers) / 1000,
        'total_apparent_power_kva': sum(apparent_powers) / 1000,
        'average_power_factor': sum(active_powers) / sum(apparent_powers),
        'number_of_loads': len(line['loads']),
        'max_load_kw': max(active_powers) / 1000,
        'min_load_kw': min(active_powers) / 1000
    }


def optimize_line_simple(
    line: Line,
    max_voltage_drop: float = 5.0,
    epochs: int = 100,
    learning_rate: float = 0.01
) -> Dict[str, Any]:
    """
    Simple optimization interface for a single line.
    
    Args:
        line: Line instance to optimize
        max_voltage_drop: Maximum allowed voltage drop percentage
        epochs: Number of optimization epochs
        learning_rate: Learning rate for optimization
        
    Returns:
        Dictionary with optimization results
    """
    from lineoptim.components import Network
    
    # Create network and optimize
    network = Network()
    network.add(line)
    
    # Get initial resistivity
    initial_resistivity = line.get_resistivity_tensor().clone()
    
    # Perform optimization
    network.optimize(
        epochs=epochs,
        lr=learning_rate,
        max_v_drop=max_voltage_drop,
        auto_stop=True
    )
    
    # Get final resistivity
    final_resistivity = line.get_resistivity_tensor()
    
    return {
        'initial_resistivity': initial_resistivity,
        'final_resistivity': final_resistivity,
        'resistivity_change': final_resistivity - initial_resistivity,
        'max_voltage_drop_target': max_voltage_drop,
        'epochs_used': epochs
    }


def create_example_network() -> Line:
    """
    Create an example network for testing and demonstration.
    
    Returns:
        Example Line instance with typical loads
    """
    # Create main line
    main_line = create_three_phase_line(
        name="Example Distribution Line",
        voltage=400.0,
        resistivity=0.15
    )
    
    # Add various loads
    add_residential_load(main_line, "House 1", 50, 5.0)
    add_residential_load(main_line, "House 2", 150, 7.2)
    add_industrial_load(main_line, "Factory", 300, 50.0)
    add_residential_load(main_line, "House 3", 450, 4.5)
    add_industrial_load(main_line, "Workshop", 600, 15.0)
    
    return main_line