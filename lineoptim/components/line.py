from collections import OrderedDict
import numpy as np
import torch
import json


def calc_apparent_power(active_power: float, power_factor: float) -> float:
    return active_power / power_factor


def calc_power_factor(active_power: float, apparent_power: float) -> float:
    return active_power / apparent_power


def find_dict_by_name(list_of_dicts, name):
    return next((item for item in list_of_dicts if item['name'] == name), None)


def get_line_names(list_of_dicts):
    return [item['name'] for item in list_of_dicts if item.get('is_line')]


def get_current(load):
    if 'active_power' in load and 'v_nominal' in load and 'power_factor' in load:
        return load['active_power'] / (load['v_nominal'] * np.sqrt(3) * load['power_factor'])
    elif len(load['loads']) > 0:
        return sum(get_current(sub_load) for sub_load in load['loads'])
    else:
        return 0.0


def tensor_to_list(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()  # Tensor in Liste umwandeln
    elif isinstance(obj, dict):
        return {key: tensor_to_list(value) for key, value in obj.items()}  # Funktion auf Wörterbuchwerte anwenden
    elif isinstance(obj, list):
        return [tensor_to_list(element) for element in obj]  # Funktion auf Listenelemente anwenden
    else:
        return obj  # Objekt zurückgeben, wenn es kein Tensor, Wörterbuch oder Liste ist


def list_to_tensor(obj):
    if isinstance(obj, list):
        if all(isinstance(i, (int, float)) for i in obj):
            return torch.tensor(obj)  # Liste in Tensor umwandeln, wenn alle Elemente Zahlen sind
        elif isinstance(obj, dict):
            return {key: list_to_tensor(value) for key, value in obj.items()}
        else:
            return obj  # Liste zurückgeben, wenn nicht alle Elemente Zahlen sind
    elif isinstance(obj, dict):
        return {key: list_to_tensor(value) for key, value in obj.items()}  # Funktion auf Wörterbuchwerte anwenden
    else:
        return obj  # Objekt zurückgeben, wenn es kein Wörterbuch oder Liste ist


def get_dUx(resistivity, reactance, loads, node_id: int = None, **kwargs):
    """
    Calculate delta U at node_id.

    Note: execute method compute_partial_voltages() before executing this method

    :param node_id: Node ID of load partial voltage to calculate. 0=first node, None=last node
    :param loads: list of loads
    :param resistivity: Conductor resistivity (ohm/m)
    :param reactance: Conductor reactance (ohm/m)
    :return: delta U at node_id
    """

    node_id = len(loads) if node_id is None else node_id + 1

    current_list = torch.stack([get_current(load) for load in loads])

    assert len(current_list), "No currents for calculation available"

    if node_id is not None:
        current_list[-1] += sum(
            load['active_power'] / (load['v_nominal'] * np.sqrt(3) * load['power_factor'])
            for load in loads[node_id:]
        )

    Mw, Mb = 0, 0

    for load, current in zip(loads[:node_id], current_list):
        position = load['position']
        power_factor = load['power_factor']
        phase_shift_angle = torch.acos(torch.tensor(power_factor))

        Mw += current * position * power_factor  # calc current momentum
        Mb += current * position * torch.tan(phase_shift_angle)  # calc current reactive moment

    resistivity = torch.tensor(resistivity) if not isinstance(resistivity, torch.Tensor) else resistivity
    reactance = torch.tensor(reactance) if not isinstance(reactance, torch.Tensor) else reactance

    conductor_impedance = (resistivity / 1000) + 1j * reactance
    current_moment = Mw - 1j * Mb

    return abs(conductor_impedance * current_moment)


def compute_partial_voltages(line, iterations: int = 2, **kwargs) -> None:
    """
    Computes all partial voltages by iterating over the nodes multiple times (param: iteration).
    Note: In case of nested line increase iterations count
    :param iterations: iterations for node partial voltages optimization
    :param line: line configuration
    :return: None
    """
    for _ in range(iterations):
        for node_id, load in enumerate(line['loads']):
            load["v_nominal"] = (line['v_nominal'] - get_dUx(node_id=node_id, **line)).detach()
            compute_partial_voltages(load, iterations=iterations) if load.get('loads') else None


class Line:
    def __init__(
            self,
            name,
            position,
            cores: OrderedDict,
            resistivity=0,
            reactance=0,
            v_nominal: torch.Tensor = torch.tensor([400.0]),
            loads=None,
            **kwargs
    ):
        """
        Electrical line instance
        :param resistivity: Conductor resistivity (ohm/m)
        :param reactance: Conductor reactance (ohm/m)
        :param v_nominal: Voltage in Volt
        :param position: Position where actual line is connected in meters
        :param kwargs:
        """
        self._idx_set_t = None
        if loads is None:
            loads = []

            assert isinstance(cores, OrderedDict), "Cores must be an OrderedDict"
            assert len(cores) > 0, "Cores must not be empty"
            assert len(cores) == len(v_nominal), "Number of cores must match number of nominal voltages"

        # init params to dict
        self._dict = {
            'name': name,
            'resistivity': resistivity,
            'reactance': reactance,
            'v_nominal': v_nominal,
            'position': position,
            'loads': loads,
            'is_line': True,
            'cores': cores
        }
        self._dict.update(kwargs)  # add additional parameters
        self._lines_to_optimize = []

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def dict(self):
        return self._dict

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __delitem__(self, key):
        del self._dict[key]

    def __iter__(self):
        return iter(self._dict['loads'])

    def __str__(self):
        return str(self._dict)

    def __repr__(self):
        return repr(self._dict)

    def __contains__(self, key):
        return key in self._dict

    def __len__(self):
        return len(self._dict['loads'])

    @property
    def loads(self):
        return self._dict['loads']

    @loads.setter
    def loads(self, value):
        self._dict['loads'] = value

    @property
    def cores(self) -> OrderedDict:
        return self._dict['cores']

    @cores.setter
    def cores(self, value: OrderedDict):
        self._dict['cores'] = value

    @property
    def voltage(self):
        """
        Voltage in Volt
        :return: Voltage list of all nodes on Line (nested lines not included)
        """
        from lineoptim.plot.accessor import VoltageAccessor
        return VoltageAccessor(self, self.cores)

    @property
    def current(self):
        """
        Conductor Current in Ampere. Loads that follow are summed up.
        :return: Current on conductor at each node on conductor (nested lines not included)
        """
        from lineoptim.plot.accessor import CurrentAccessor
        return CurrentAccessor(self, self.cores)

    @property
    def current_sum(self):
        """
        Conductor Current in Ampere. Loads that follow are summed up.
        :return: Current on conductor at each node on conductor (nested lines not included)
        """
        from lineoptim.plot.accessor import SumCurrentAccessor
        return SumCurrentAccessor(self, self.cores)

    @property
    def voltage_unbalance(self):
        """
        Voltage unbalance in % using NEMA definition
        :return: Voltage unbalance list of all nodes on Line (nested lines not included)
        """
        from lineoptim.plot.accessor import VoltageUnbalanceAccessor
        return VoltageUnbalanceAccessor(self, self.cores)

    @property
    def current_unbalance(self):
        """
        Current unbalance in % using NEMA definition
        :return: Current unbalance list of all nodes on Line (nested lines not included)
        """
        from lineoptim.plot.accessor import CurrentUnbalanceAccessor
        return CurrentUnbalanceAccessor(self, self.cores)

    def recompute(self, iterations=2) -> None:
        """
        Recompute partial voltages by taking all nodes and nested lines into account
        :param iterations: Number of iterations, increase for more accurate results
        :return: None
        """
        compute_partial_voltages(self, iterations=iterations)

    def get_resistivity_tensor(self) -> torch.Tensor:
        """
        Get resistivity tensor representing resistivity of each core for each load
        Note: Resistivity tensor is a 2D tensor with shape (n_loads, n_cores)
        """
        lines = [self._dict['resistivity']]  # get resistivity of actual line
        lines.extend(self._get_lines_resistivity(self._dict))  # get resistivity of nested lines

        return torch.stack(lines)  # stack resistivity to tensor

    def get_residual_voltage_tensor(self):
        """ Get residual voltage tensor on all lines and nested lines """

        return torch.stack(self._get_lines_udx(self._dict))

    def set_resistivity_tensor(self, resistivity: torch.Tensor) -> None:
        """ Set resistivity tensor representing resistivity of each core for each load """
        self._idx_set_t = 0
        self._dict['resistivity'] = resistivity[self._idx_set_t]  # set resistivity of actual line
        self._set_lines_resistivity(self._dict, resistivity)  # set resistivity of nested lines

    def _set_lines_resistivity(self, line, resistivity):
        for load in line['loads']:
            if load.get('is_line'):
                self._idx_set_t += 1
                load['resistivity'] = resistivity[self._idx_set_t]
                self._set_lines_resistivity(load, resistivity)

    def _get_lines_resistivity(self, line):
        lines = []
        for load in line['loads']:
            if load.get('is_line'):
                lines.append(load['resistivity'])
                lines.extend(self._get_lines_resistivity(load))
        return lines

    def _get_lines_udx(self, line):
        lines = [line['v_nominal'] - get_dUx(**line)]
        for load in line['loads']:
            if load.get('is_line'):
                lines.extend(self._get_lines_udx(load))
        return lines

    def add(self, name: str, position: float, **kwargs) -> None:
        """
        Add load to line
        Note: Loads will automatically be sorted by position.
        :param name: Name of load
        :param position: Position of load in meters
        :param kwargs: Additional parameters
        :return: None
        """
        load = {'name': name, 'position': position, **kwargs}
        if 'active_power' in load and 'power_factor' in load:
            load['apparent_power'] = load['active_power'] / load['power_factor']  # calc apparent power
            load['is_line'] = False  # set is_line to False
        elif len(load['loads']) > 0:  # check if it is a line
            load['active_power'] = sum(load['active_power'] for load in load['loads'])  # sum active power
            load['apparent_power'] = sum(load['apparent_power'] for load in load['loads'])  # sum apparent power
            load['power_factor'] = load['active_power'] / load['apparent_power']  # calc power factor
            load['is_line'] = True  # set is_line to True
        else:
            raise ValueError('No active power and power factor defined. Check your load definition.')

        load['cores'] = load.get('cores', None)  # add cores information
        self._dict['loads'].append(load)
        self._dict['loads'] = sorted(self._dict['loads'], key=lambda x: x['position'])  # sort loads by position

        for idx, load in enumerate(self._dict['loads']):
            load['idx'] = idx  # add index to load

    def get_current_by_idx(self, idx: int):
        """
        Calculate current by idx.
        Note: Current corresponds to the current of selected node_id.
        :param idx: Load index
        :return: Current in Ampere
        """
        return get_current(self._dict['loads'][idx])

    def get_spot_current(self, idx=0):
        return sum(get_current(load) for load in self._dict['loads'][idx:])

    def get_total_current(self):
        return self.get_spot_current()

    def save_to_json(self, filename: str):
        with open(filename, 'w') as f:
            json.dump(tensor_to_list(self._dict), f, indent=4)

    def load_from_json(self, filename: str):
        with open(filename, 'r') as f:
            self._dict = json.load(f)
