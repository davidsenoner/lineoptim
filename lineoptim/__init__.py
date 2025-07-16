"""
LineOptim: Electric Power Line Parameter Optimization

A comprehensive Python package for electric power line simulation and optimization.
"""

__version__ = "1.2.1"
__author__ = "David Senoner"
__email__ = "david.senoner@gmail.com"

# Core components
from lineoptim.components.line import Line, compute_partial_voltages, get_dUx, get_current
from lineoptim.components.network import Network

# Visualization
from lineoptim.plot import PlotGraph

# Utilities
from lineoptim.components.line import (
    calc_apparent_power,
    calc_power_factor,
    tensor_to_list,
    list_to_tensor
)

# High-level utilities
from lineoptim.utils import (
    create_three_phase_line,
    create_single_phase_line,
    add_residential_load,
    add_industrial_load,
    calculate_voltage_drop_percentage,
    get_power_summary,
    optimize_line_simple,
    create_example_network
)

__all__ = [
    # Core classes
    "Line",
    "Network",
    "PlotGraph",
    
    # Core functions
    "compute_partial_voltages",
    "get_dUx",
    "get_current",
    
    # Utilities
    "calc_apparent_power",
    "calc_power_factor",
    "tensor_to_list",
    "list_to_tensor",
    
    # High-level utilities
    "create_three_phase_line",
    "create_single_phase_line",
    "add_residential_load",
    "add_industrial_load",
    "calculate_voltage_drop_percentage",
    "get_power_summary",
    "optimize_line_simple",
    "create_example_network",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
]
