import numpy as np
import torch
import json


def calc_apparent_power(active_power: float, power_factor: float) -> float:
    return active_power / power_factor


def calc_power_factor(active_power: float, apparent_power: float) -> float:
    return active_power / apparent_power


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


class Line:
    def __init__(
            self,
            name,
            position,
            resistivity=0,
            reactance=0,
            v_nominal=400,
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
        if loads is None:
            loads = []

        # init params to dict
        self._dict = {
            'name': name,
            'resistivity': resistivity,
            'reactance': reactance,
            'v_nominal': v_nominal,
            'position': position,
            'loads': loads
        }
        self._dict.update(kwargs)  # add additional parameters

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
        return iter(self._dict)

    def __str__(self):
        return str(self._dict)

    def __repr__(self):
        return repr(self._dict)

    def __contains__(self, key):
        return key in self._dict

    @property
    def loads(self):
        return self._dict['loads']

    @loads.setter
    def loads(self, value):
        self._dict['loads'] = value

    def get_load_len(self) -> int:
        return len(self.loads)

    def get_load(self, idx: int):
        return self.loads[idx]

    def add(self, name: str, position: float, **kwargs) -> None:
        """
        Add load to line
        Note: Loads will automatically be sorted by position.
        :param name: Name of load
        :param position: Position of load in meters
        :param kwargs: Additional parameters
        :return: None
        """
        load = {
            'name': name,
            'position': position,
        }
        load.update(kwargs)  # add additional parameters

        if 'active_power' in load and 'power_factor' in load:
            load['apparent_power'] = load['active_power'] / load['power_factor']
        elif len(load['loads']) > 0:
            load['active_power'] = sum(load['active_power'] for load in load['loads'])
            load['apparent_power'] = sum(load['apparent_power'] for load in load['loads'])
            load['power_factor'] = load['active_power'] / load['apparent_power']
        else:
            raise ValueError('No active power and power factor defined. Check your load definition.')
        self.loads.append(load)  # add load to line
        self.loads = sorted(self.loads, key=lambda x: x['position'])  # sort loads by position
        # assign idx to loads
        for idx, load in enumerate(self.loads):
            load['idx'] = idx

    @staticmethod
    def calc_nested_currents(loads):
        """
        Calculate current by summing up all currents of line loads and all nested loads.
        :param loads: list of loads
        :return: Current in Ampere
        """
        current = 0.0
        for load in loads:
            if not isinstance(load['v_nominal'], torch.Tensor):
                load['v_nominal'] = torch.tensor(load['v_nominal'])
            if 'active_power' in load and 'v_nominal' in load and 'power_factor' in load:
                current += load['active_power'] / (load['v_nominal'] * np.sqrt(3) * load['power_factor'])
            elif len(load['loads']) > 0:
                current += Line.calc_nested_currents(load['loads'])
        return current

    @staticmethod
    def get_current(load):
        """
        Calculate current at node_id.
        Note: Current corresponds to the current of selected node_id.
        :param load: Load configuration
        :return: Current in Ampere
        """

        # if load is a single load
        if 'active_power' in load and 'v_nominal' in load and 'power_factor' in load:
            if not isinstance(load['v_nominal'], torch.Tensor):
                load['v_nominal'] = torch.tensor(load['v_nominal'])
            return load['active_power'] / (load['v_nominal'] * np.sqrt(3) * load['power_factor'])

        # if load has nested loads (sub-line)
        elif len(load['loads']) > 0:
            current = 0.0
            for load in load['loads']:
                # convert v_nominal to tensor
                if not isinstance(load['v_nominal'], torch.Tensor):
                    load['v_nominal'] = torch.tensor(load['v_nominal'])

                # calculate current if it is a single load
                if 'active_power' in load and 'v_nominal' in load and 'power_factor' in load:
                    current += load['active_power'] / (load['v_nominal'] * np.sqrt(3) * load['power_factor'])

                # calculate current if it is a sub-line
                elif len(load['loads']) > 0:
                    current += Line.get_current(load)
            return current
        else:
            return 0.0

    def get_current_by_idx(self, idx: int):
        """
        Calculate current by idx.
        Note: Current corresponds to the current of selected node_id.
        :param idx: Load index
        :return: Current in Ampere
        """
        return Line.get_current(self.loads[idx])

    def get_current_at_idx(self, idx=0):
        """
        Calculate current at line node_id.
        Note: Current corresponds to the current of selected node_id and all following nodes.
        :param idx: Node ID to calulate current. 0=first node
        :return: Current in Ampere
        """
        return Line.calc_nested_currents(self.loads[idx:])

    @staticmethod
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

        current_list = torch.stack([Line.get_current(load) for load in loads])

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

    @staticmethod
    def compute_partial_voltages(line, iterations: int = 2, **kwargs) -> None:
        """
        Computes all partial voltages by iterating over the nodes multiple times (param: iteration).
        Note: In case of nested line increase iterations count
        :param iterations: iterations for node partial voltages optimization
        :param line: line configuration
        :return: None
        """

        # recompute partial voltages
        for _ in range(iterations):
            for node_id, load in enumerate(line['loads']):
                drop_voltage = line['v_nominal'] - Line.get_dUx(node_id=node_id, **line)
                load["v_nominal"] = drop_voltage.detach()

                if load.get('loads'):
                    Line.compute_partial_voltages(load, iterations=iterations)

    def save_to_json(self, filename: str):
        with open(filename, 'w') as f:
            json.dump(tensor_to_list(self._dict), f, indent=4)

    def load_from_json(self, filename: str):
        with open(filename, 'r') as f:
            self._dict = json.load(f)
