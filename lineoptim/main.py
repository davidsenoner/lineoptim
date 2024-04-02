import torch

from lineoptim.components import Network, Line

LINE_CFG = 'resources/example_line.json'  # network configuration file

if __name__ == '__main__':

    v_nominal = torch.tensor([400.0, 400.0, 400.0])  # nominal voltage

    line_params = {
        "name": "Main power line 1",
        "position": 0,
        "resistivity": torch.tensor([0.125, 0.125, 0.125]),  # resistivity,
        "reactance": torch.tensor([0.0, 0.0, 0.0]),
        "v_nominal": v_nominal,
    }

    main_line = Line(**line_params)  # create main line instance

    main_line.add("Load 1", 100, active_power=20000, v_nominal=v_nominal, power_factor=0.9)
    main_line.add("Load 2", 200, active_power=20000, v_nominal=v_nominal, power_factor=0.9)
    main_line.add("Load 3", 300, active_power=20000, v_nominal=v_nominal, power_factor=0.9)
    main_line.add("Load 4", 400, active_power=20000, v_nominal=v_nominal, power_factor=0.9)

    line = Line('Sub-line 1', 500, v_nominal=v_nominal, resistivity=torch.tensor([0.125, 0.125, 0.125]))  # create line instance

    line.add("Load 1.1", 100, active_power=2000, v_nominal=v_nominal, power_factor=0.9)
    line.add("Load 1.2", 150, active_power=2000, v_nominal=v_nominal, power_factor=0.9)
    line.add("Load 1.3", 200, active_power=2000, v_nominal=v_nominal, power_factor=0.9)
    line.add("Load 1.4", 250, active_power=2000, v_nominal=v_nominal, power_factor=0.9)

    main_line.add(**line.dict())  # add line to main line

    main_line.add("Load 5", 600, active_power=10000, v_nominal=v_nominal, power_factor=0.9)
    main_line.add("Load 6", 700, active_power=10000, v_nominal=v_nominal, power_factor=0.9)

    print(main_line.get_current_at_idx(2))
    print(Line.get_current(main_line.loads[2]))
    print(f"Udx: {Line.get_dUx(node_id=4, **main_line.dict())}")
    Line.compute_partial_voltages(main_line.dict(), iterations=5)
    print(f"Udx: {Line.get_dUx(node_id=4, **main_line.dict())}")

    main_line.save_to_json(LINE_CFG)  # save line configuration as json
