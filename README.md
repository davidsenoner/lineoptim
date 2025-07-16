# LineOptim - Electric Power Line Parameter Optimization

[![PyPI version](https://badge.fury.io/py/lineoptim.svg)](https://badge.fury.io/py/lineoptim) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LineOptim** is an open-source Python package for electric power line simulation and optimization.
It provides comprehensive tools for simulating, visualizing, and optimizing electric power lines and is designed to scale with larger and nested networks.

## Features

- **Line Simulation**: Simulate electric power lines with multiple loads and sub-lines. Calculate voltage drop, current, and power at any point in the line.
- **Optimization**: Optimize electric power lines for desired voltage drop and optimal conductor sizing.
- **Visualization**: Visualize electric power lines and their parameters with comprehensive plotting capabilities.
- **Nested Networks**: Support for complex, nested line configurations.
- **PyTorch Integration**: Leverage PyTorch for efficient computations and automatic differentiation.


## Installation

Install LineOptim using pip:

```bash
pip install lineoptim
```

## Quick Start

```python
import torch
from collections import OrderedDict
import lineoptim as lo

# Define voltage and core configuration
v_nominal = torch.tensor([400.0, 400.0, 400.0])  # Three-phase nominal voltage
cores = OrderedDict({"L1": "brown", "L2": "black", "L3": "grey"})

# Create main power line
main_line = lo.Line(
    name="Main power line",
    position=0,
    resistivity=torch.tensor([0.15, 0.251, 0.351]),  # Ohm/km per phase
    reactance=torch.tensor([0.0, 0.0, 0.0]),
    v_nominal=v_nominal,
    cores=cores
)

# Add loads to the line
main_line.add("Load 1", 100, active_power=2000, v_nominal=v_nominal, power_factor=0.91)
main_line.add("Load 2", 500, active_power=5000, v_nominal=v_nominal, power_factor=0.85)

# Compute electrical parameters
main_line.recompute()

# Access results
print("Voltages:", main_line.voltage)
print("Currents:", main_line.current)
```

## Advanced Usage

### Network Optimization

```python
# Create network and optimize conductor sizing
network = lo.Network()
network.add(main_line)
network.optimize(epochs=200, lr=0.01, max_v_drop=5.0)
```

### Visualization

```python
# Plot network graph
graph = lo.PlotGraph(main_line)
graph.plot()

# Plot voltage and current curves
main_line.voltage.plot()
main_line.current.plot()
```

For more detailed examples, check the `examples/` directory.

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Bug Reports**: Use the [GitHub Issue Tracker](https://github.com/davidsenoner/lineoptim/issues) to report bugs.
2. **Feature Requests**: Submit feature requests via GitHub Issues.
3. **Pull Requests**: Fork the repository and submit pull requests for improvements.
4. **Code Style**: Follow PEP 8 and include appropriate tests.

## Development

To set up the development environment:

```bash
git clone https://github.com/davidsenoner/lineoptim.git
cd lineoptim
pip install -e .
```

## License

**LineOptim** is licensed under the [MIT License](https://github.com/davidsenoner/lineoptim/blob/main/LICENCE).

## Citation

If you use LineOptim in your research, please cite:

```bibtex
@software{lineoptim,
  title={LineOptim: Electric Power Line Parameter Optimization},
  author={David Senoner},
  url={https://github.com/davidsenoner/lineoptim},
  year={2024}
}
```

