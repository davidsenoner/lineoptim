from collections import OrderedDict
from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch
import json


def calc_apparent_power(active_power: float, power_factor: float) -> float:
    """
    Calculate apparent power from active power and power factor.
    
    Args:
        active_power: Active power in Watts
        power_factor: Power factor (0 < power_factor <= 1)
        
    Returns:
        Apparent power in VA
        
    Raises:
        ValueError: If power_factor is not in valid range
    """
    if not (0 < power_factor <= 1):
        raise ValueError("Power factor must be between 0 and 1")
    return active_power / power_factor


def calc_power_factor(active_power: float, apparent_power: float) -> float:
    """
    Calculate power factor from active and apparent power.
    
    Args:
        active_power: Active power in Watts
        apparent_power: Apparent power in VA
        
    Returns:
        Power factor (0 < power_factor <= 1)
        
    Raises:
        ValueError: If apparent_power is zero or negative
    """
    if apparent_power <= 0:
        raise ValueError("Apparent power must be positive")
    return active_power / apparent_power


def find_dict_by_name(list_of_dicts: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    """
    Find dictionary by name in a list of dictionaries.
    
    Args:
        list_of_dicts: List of dictionaries to search
        name: Name to search for
        
    Returns:
        Dictionary with matching name or None if not found
    """
    return next((item for item in list_of_dicts if item.get('name') == name), None)


def get_line_names(list_of_dicts: List[Dict[str, Any]]) -> List[str]:
    """
    Get names of all lines from a list of dictionaries.
    
    Args:
        list_of_dicts: List of dictionaries representing loads/lines
        
    Returns:
        List of line names
    """
    return [item['name'] for item in list_of_dicts if item.get('is_line')]


def get_current(load: Dict[str, Any]) -> Union[float, torch.Tensor]:
    """
    Calculate current for a load or sum of currents for nested loads.
    
    Args:
        load: Dictionary representing a load or line with nested loads
        
    Returns:
        Current in Amperes
    """
    if 'active_power' in load and 'v_nominal' in load and 'power_factor' in load:
        return load['active_power'] / (load['v_nominal'] * np.sqrt(3) * load['power_factor'])
    elif 'loads' in load and len(load['loads']) > 0:
        return sum(get_current(sub_load) for sub_load in load['loads'])
    else:
        return 0.0


def tensor_to_list(obj: Any) -> Any:
    """
    Recursively convert PyTorch tensors to lists for JSON serialization.
    
    Args:
        obj: Object that may contain tensors
        
    Returns:
        Object with tensors converted to lists
    """
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: tensor_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_list(element) for element in obj]
    else:
        return obj


def list_to_tensor(obj: Any) -> Any:
    """
    Recursively convert lists to PyTorch tensors where appropriate.
    
    Args:
        obj: Object that may contain lists to convert
        
    Returns:
        Object with appropriate lists converted to tensors
    """
    if isinstance(obj, list) and all(isinstance(i, (int, float)) for i in obj):
        return torch.tensor(obj)
    elif isinstance(obj, dict):
        return {key: list_to_tensor(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [list_to_tensor(element) for element in obj]
    else:
        return obj


def get_dUx(
    resistivity: Union[torch.Tensor, List[float]], 
    reactance: Union[torch.Tensor, List[float]], 
    loads: List[Dict[str, Any]], 
    node_id: Optional[int] = None, 
    **kwargs
) -> torch.Tensor:
    """
    Calculate voltage drop (delta U) at specified node.

    Args:
        resistivity: Conductor resistivity in ohm/m per phase
        reactance: Conductor reactance in ohm/m per phase
        loads: List of load dictionaries
        node_id: Node ID for voltage calculation (0=first node, None=last node)
        **kwargs: Additional parameters
        
    Returns:
        Voltage drop at specified node
        
    Raises:
        AssertionError: If no currents available for calculation
        
    Note:
        Execute compute_partial_voltages() before calling this method
    """
    node_id = len(loads) if node_id is None else node_id + 1

    current_list = torch.stack([get_current(load) for load in loads])

    assert len(current_list) > 0, "No currents for calculation available"

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


def compute_partial_voltages(line: Dict[str, Any], iterations: int = 2, **kwargs) -> None:
    """
    Compute partial voltages for all nodes by iterating multiple times.
    
    Args:
        line: Line configuration dictionary
        iterations: Number of iterations for voltage optimization (increase for nested lines)
        **kwargs: Additional parameters
        
    Returns:
        None (modifies line dictionary in-place)
        
    Note:
        For nested lines, increase the iterations count for better accuracy
    """
    for _ in range(iterations):
        for node_id, load in enumerate(line['loads']):
            load["v_nominal"] = (line['v_nominal'] - get_dUx(node_id=node_id, **line)).detach()
            if load.get('loads'):
                compute_partial_voltages(load, iterations=iterations)


class Line:
    """
    Electrical power line representation with loads and nested lines support.
    
    This class represents an electrical power line with support for multiple loads,
    nested sub-lines, and comprehensive electrical calculations.
    """
    
    def __init__(
        self,
        name: str,
        position: float,
        cores: OrderedDict,
        resistivity: Union[float, torch.Tensor] = 0,
        reactance: Union[float, torch.Tensor] = 0,
        v_nominal: torch.Tensor = torch.tensor([400.0]),
        loads: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Initialize electrical line instance.
        
        Args:
            name: Name of the line
            position: Position where line is connected (meters)
            cores: Ordered dictionary of core names and colors
            resistivity: Conductor resistivity in ohm/m per phase
            reactance: Conductor reactance in ohm/m per phase
            v_nominal: Nominal voltage per phase in Volts
            loads: List of loads connected to this line
            **kwargs: Additional parameters
            
        Raises:
            AssertionError: If validation fails for cores or voltage parameters
        """
        self._idx_set_t = None
        if loads is None:
            loads = []

        # Validation
        assert isinstance(cores, OrderedDict), "Cores must be an OrderedDict"
        assert len(cores) > 0, "Cores must not be empty"
        assert len(cores) == len(v_nominal), "Number of cores must match number of nominal voltages"

        # Initialize parameters dictionary
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
        self._dict.update(kwargs)  # Add additional parameters
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
        from lineoptim.plot.plotter import VoltagePlotter
        return VoltagePlotter(self, self.cores)

    @property
    def current(self):
        """
        Conductor Current in Ampere. Loads that follow are summed up.
        :return: Current on conductor at each node on conductor (nested lines not included)
        """
        from lineoptim.plot.plotter import CurrentPlotter
        return CurrentPlotter(self, self.cores)

    @property
    def current_sum(self):
        """
        Conductor Current in Ampere. Loads that follow are summed up.
        :return: Current on conductor at each node on conductor (nested lines not included)
        """
        from lineoptim.plot.plotter import SumCurrentPlotter
        return SumCurrentPlotter(self, self.cores)

    @property
    def voltage_unbalance(self):
        """
        Voltage unbalance in % using NEMA definition
        :return: Voltage unbalance list of all nodes on Line (nested lines not included)
        """
        from lineoptim.plot.plotter import VoltageUnbalancePlotter
        return VoltageUnbalancePlotter(self, self.cores)

    @property
    def current_unbalance(self):
        """
        Current unbalance in % using NEMA definition
        :return: Current unbalance list of all nodes on Line (nested lines not included)
        """
        from lineoptim.plot.plotter import CurrentUnbalancePlotter
        return CurrentUnbalancePlotter(self, self.cores)

    @property
    def apparent_power(self):
        """
        Apparent power in VA
        :return: Apparent power list of all nodes on Line (nested lines not included)
        """
        from lineoptim.plot.plotter import ApparentPowerPlotter
        return ApparentPowerPlotter(self, self.cores)

    @property
    def active_power(self):
        """
        Active power in W
        :return: Active power list of all nodes on Line (nested lines not included)
        """
        from lineoptim.plot.plotter import ActivePowerPlotter
        return ActivePowerPlotter(self, self.cores)

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
        Add load or sub-line to the line.
        
        Args:
            name: Name of the load or sub-line
            position: Position along the line in meters
            **kwargs: Additional parameters including:
                - active_power: Active power in Watts
                - power_factor: Power factor (0 < power_factor <= 1)
                - v_nominal: Nominal voltage
                - loads: List of nested loads (for sub-lines)
                
        Raises:
            ValueError: If required parameters are missing or invalid
            
        Note:
            Loads are automatically sorted by position after addition.
        """
        load = {'name': name, 'position': position, **kwargs}
        
        # Validate and process load parameters
        if 'active_power' in load and 'power_factor' in load:
            if load['power_factor'] <= 0 or load['power_factor'] > 1:
                raise ValueError("Power factor must be between 0 and 1")
            load['apparent_power'] = load['active_power'] / load['power_factor']
            load['is_line'] = False
        elif 'loads' in load and len(load['loads']) > 0:
            # Calculate aggregated power for sub-line
            load['active_power'] = sum(sub_load.get('active_power', 0) for sub_load in load['loads'])
            load['apparent_power'] = sum(sub_load.get('apparent_power', 0) for sub_load in load['loads'])
            if load['apparent_power'] > 0:
                load['power_factor'] = load['active_power'] / load['apparent_power']
            else:
                load['power_factor'] = 1.0
            load['is_line'] = True
        else:
            raise ValueError('Either active_power and power_factor, or loads must be specified')

        load['cores'] = load.get('cores', None)
        self._dict['loads'].append(load)
        
        # Sort loads by position and assign indices
        self._dict['loads'] = sorted(self._dict['loads'], key=lambda x: x['position'])
        for idx, load in enumerate(self._dict['loads']):
            load['idx'] = idx

    def get_spot_current(self, idx=0):
        return sum(get_current(load) for load in self._dict['loads'][idx:])

    def get_total_current(self):
        return self.get_spot_current()

    def save_to_json(self, filename: str) -> None:
        """
        Save line configuration to JSON file.
        
        Args:
            filename: Path to output JSON file
            
        Raises:
            IOError: If file cannot be written
        """
        try:
            with open(filename, 'w') as f:
                json.dump(tensor_to_list(self._dict), f, indent=4)
        except IOError as e:
            raise IOError(f"Cannot write to file {filename}: {e}")

    def load_from_json(self, filename: str) -> None:
        """
        Load line configuration from JSON file.
        
        Args:
            filename: Path to input JSON file
            
        Raises:
            IOError: If file cannot be read
            json.JSONDecodeError: If file is not valid JSON
        """
        try:
            with open(filename, 'r') as f:
                self._dict = json.load(f)
        except IOError as e:
            raise IOError(f"Cannot read file {filename}: {e}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON in file {filename}: {e}", e.doc, e.pos)
