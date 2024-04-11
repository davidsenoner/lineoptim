import torch

from lineoptim.components import Network, Line
from lineoptim.components.line import compute_partial_voltages, get_dUx, get_current

LINE_CFG = 'resources/example_line.json'  # network configuration file

if __name__ == '__main__':

    v_nominal = torch.tensor([400.0, 400.0, 400.0])  # nominal voltage

    line_params = {
        "name": "Main power line 1",
        "position": 0,
        "resistivity": torch.tensor([0.12, 0.12, 0.12]),  # resistivity,
        "reactance": torch.tensor([0.0, 0.0, 0.0]),
        "v_nominal": v_nominal,
    }

    main_line = Line(**line_params)  # create main line instance

    main_line.add("Load 1", 100, active_power=20000, v_nominal=v_nominal, power_factor=0.9)
    main_line.add("Load 2", 200, active_power=20000, v_nominal=v_nominal, power_factor=0.9)
    main_line.add("Load 3", 300, active_power=20000, v_nominal=v_nominal, power_factor=0.9)
    main_line.add("Load 4", 400, active_power=20000, v_nominal=v_nominal, power_factor=0.9)
    main_line.add("Load 5", 500, active_power=20000, v_nominal=v_nominal, power_factor=0.9)
    main_line.add("Load 6", 600, active_power=20000, v_nominal=v_nominal, power_factor=0.9)
    main_line.add("Load 7", 700, active_power=20000, v_nominal=v_nominal, power_factor=0.9)
    main_line.add("Load 8", 800, active_power=20000, v_nominal=v_nominal, power_factor=0.9)
    main_line.add("Load 9", 900, active_power=20000, v_nominal=v_nominal, power_factor=0.9)
    main_line.add("Load 10", 1000, active_power=20000, v_nominal=v_nominal, power_factor=0.9)


    line = Line('Sub-line 1', 550, v_nominal=v_nominal, resistivity=torch.tensor([0.145, 0.146, 0.147]))
    line.add("Load 1.1", 100, active_power=20000, v_nominal=v_nominal, power_factor=0.9)
    line.add("Load 1.2", 150, active_power=20000, v_nominal=v_nominal, power_factor=0.9)
    line.add("Load 1.3", 200, active_power=20000, v_nominal=v_nominal, power_factor=0.9)
    line.add("Load 1.4", 250, active_power=20000, v_nominal=v_nominal, power_factor=0.9)

    line1 = Line('Sub-line 2', 500, v_nominal=v_nominal, resistivity=torch.tensor([0.135, 0.136, 0.137]))# create line instance
    line1.add("Load 2.3", 200, active_power=2000, v_nominal=v_nominal, power_factor=0.9)
    line1.add("Load 2.4", 250, active_power=2000, v_nominal=v_nominal, power_factor=0.9)

    #line.add(**line1.dict())  # add line to main line
    main_line.add(**line.dict())  # add line to main line

    main_line.add("Load 5", 1100, active_power=10000, v_nominal=v_nominal, power_factor=0.9)
    main_line.add("Load 6", 1200, active_power=10000, v_nominal=v_nominal, power_factor=0.9)

    print(f"Spot current (get_spot_current): {main_line.get_spot_current(2)}")
    print(f"Load current (get_current_by_idx): {main_line.get_current_by_idx(2)}")

    print(f"Load current (get_current): {get_current(main_line.loads[2])}")
    print(f"Udx: {get_dUx(node_id=4, **main_line.dict())}")
    compute_partial_voltages(main_line, iterations=5)
    print(f"Udx (computed): {get_dUx(node_id=4, **main_line.dict())}")

    print(f"Lines resistivity to optimize: {main_line.get_resistivity_tensor()}")
    print(f"Cores: {main_line.cores_to_optimize()}")
    print(f"Loads: {main_line.loads_to_optimize()}")

    print(f"Residual voltage Tensor: {main_line.get_residual_voltage_tensor()}")

    network = Network()
    network.add(main_line)

    network.optimize(epochs=200, lr=0.01, max_v_drop=5.0)  # optimize network

    main_line.save_to_json(LINE_CFG)  # save line configuration as json