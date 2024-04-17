# LineOptim - Electric Power Line Parameter Optimization

[![PyPI version](https://badge.fury.io/py/lineoptim.svg)](https://badge.fury.io/py/lineoptim) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**lineoptim** is an open-source package for electric power line simulation and optimization.
It provides the tools for simulating, visualizing and optimizing electric power lines and is designed to scale with larger and nested networks.

## Features

- **Line Simulation**: Simulate electric power lines with multiple loads and sub-lines. Get voltage drop, current power at any point in the line.
- **Optimization**: Optimize electric power lines for wanted voltage drop and optimal conductor size
- **Visualization**: Visualize electric power lines and their parameters


## Installation
```bash
pip install lineoptim
```

## Usage
```python
import lineoptim as lo
import torch

# Create a line
v_nominal = torch.tensor([400.0, 400.0, 400.0])  # nominal voltage

line_params = {
    "name": "Main power line 1",
    "position": 0,
    "resistivity": torch.tensor([0.12, 0.12, 0.12]),  # resistivity,
    "reactance": torch.tensor([0.0, 0.0, 0.0]),
    "v_nominal": v_nominal,
}

main_line = lo.Line(**line_params)

main_line.add("Load 1", 100, active_power=20000, v_nominal=v_nominal, power_factor=0.9)
main_line.add("Load 2", 200, active_power=20000, v_nominal=v_nominal, power_factor=0.9)
main_line.add("Load 3", 300, active_power=20000, v_nominal=v_nominal, power_factor=0.9)

sub_line = lo.Line('Sub-line 1', 400, v_nominal=v_nominal, resistivity=torch.tensor([0.145, 0.145, 0.145]))

main_line.add(**sub_line.dict())

network = lo.Network()  # create network
network.add(main_line)  # add line to network

network.optimize(epochs=200, lr=0.01, max_v_drop=5.0)  # optimize network on 5% voltage drop at line ends

main_line.save_to_json('resources/example_line.json')  # save line configuration as json
```

## Contributing
Contributions are welcome! For feature requests, bug reports or submitting pull requests, please use the [GitHub Issue Tracker](https://github.com/davidsenoner/lineoptim/issues).

# License
**lineoptim** is licensed under the open source [MIT Licence](https://github.com/davidsenoner/lineoptim/blob/main/LICENCE)

