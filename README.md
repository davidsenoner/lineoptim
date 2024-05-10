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

Check examples folder for more examples.

```python
import torch
from collections import OrderedDict
from matplotlib import pyplot as plt

import lineoptim as lo

LINE_CFG = 'example_line.json'  # network configuration file

if __name__ == '__main__':
    v_nominal = torch.tensor([400.0, 400.0, 400.0])  # nominal voltage
    cores = OrderedDict({"L1": "brown", "L2": "black", "L3": "grey"})

    line_params = {
        "name": "Main power line 1",
        "position": 0,
        "resistivity": torch.tensor([0.15, 0.251, 0.351]),  # resistivity,
        "reactance": torch.tensor([0.0, 0.0, 0.0]),
        "v_nominal": v_nominal,
        "cores": cores
    }

    main_line = lo.Line(**line_params)  # create main line instance

    # add some loads on main line
    main_line.add("Load 1", 100, active_power=2000, v_nominal=v_nominal, power_factor=0.91)
    main_line.add("Load 9", 900, active_power=6000, v_nominal=v_nominal, power_factor=0.9)

    # create a subline
    line = lo.Line('Sub-line 1', 300, v_nominal=v_nominal, resistivity=torch.tensor([0.2, 0.2, 0.2]), cores=cores)
    line.add("Load 1.1", 100, active_power=20000, v_nominal=v_nominal, power_factor=0.8)

    main_line.add(**line.dict())  # add line to main line

    # compute load currents
    main_line.recompute()

    # print some results
    print("Voltages on L1, L2, L3: \n", main_line.voltage)
    print("Voltage of specific load (3. load): \n", main_line.voltage[2])
    print("Currents on L1, L2, L3: \n", main_line.current)
    print("Current of specific loads (first 5 loads): \n", main_line.current[0:5])

    def plot_results(plot_line, title):
        # plot some results
        plt.style.use('bmh')
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(title)

        # Adjust the spacing between the subplots
        plt.subplots_adjust(hspace=0.3)

        plot_line.voltage.plot(ax=ax[0, 0])
        plot_line.current.plot(ax=ax[0, 1])
        plot_line.current_sum.plot(ax=ax[1, 0])
        plot_line.voltage_unbalance.plot(ax=ax[1, 1])

        fig.show()

    plot_results(main_line, "Main line before optimization")

    # optimize conductor resistivity
    network = lo.Network()  # create network instance
    network.add(main_line)  # add main line to network
    network.optimize(epochs=200, lr=0.01, max_v_drop=5.0)  # optimize network

    plot_results(main_line, "Main line after optimization")

    graph = lo.PlotGraph(main_line)
    graph.plot()  # plot network graph

    main_line.save_to_json(LINE_CFG)  # save line configuration as json
```

## Contributing
Contributions are welcome! For feature requests, bug reports or submitting pull requests, please use the [GitHub Issue Tracker](https://github.com/davidsenoner/lineoptim/issues).

# License
**lineoptim** is licensed under the open source [MIT Licence](https://github.com/davidsenoner/lineoptim/blob/main/LICENCE)

