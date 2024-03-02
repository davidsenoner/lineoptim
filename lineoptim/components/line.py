import numpy as np
from numpy.typing import NDArray
import torch
from torch import tensor, Tensor
from torch import nn
from torch.optim import Adam


class Line:
    def __init__(self,
                 resistivity: float | NDArray | Tensor = 0,
                 reactance: float | NDArray | Tensor = 0,
                 ue_voltage: float | NDArray | Tensor = 400,
                 power_factor: float = 0.9,
                 length: float = 0,
                 is_main_line: bool = False,
                 level: int = 0,
                 **kwargs
                 ):

        self._loads = []
        self._voltage = tensor(ue_voltage)  # Nominal voltage network
        self._resistivity = tensor(resistivity)
        self._reactance = tensor(reactance)
        self._power_factor = tensor([power_factor])
        self._length = length
        self._is_main_line = is_main_line
        self._level = level

    def __str__(self):
        return f'Main Line ({self.level})' if self.is_main_line else f'Sub Line ({self.level})'

    @property
    def is_main_line(self) -> bool:
        return self._is_main_line

    @property
    def level(self) -> int:
        return self._level

    @property
    def loads(self):
        return self._loads

    def get_resistivity(self) -> float | Tensor:
        """
        Cable resistivity (ohm/m)
        :return:
        """
        return self._resistivity

    def set_resistivity(self, resistance: float | Tensor) -> None:
        self._resistivity = resistance

    def get_length(self) -> float:
        """
        Distance between actual node and precedent node
        :return: Distance in meters
        """
        return self._length

    def add(self, load) -> None:
        """
        Add a node to Network

        :param load: Node
        :return: None
        """
        self._loads.append(load)

    def get_load_count(self) -> int:
        """
        Returns load quantity on line.
        Note: sub-lines are counted as one load!
        :return:
        """
        return len(self._loads)

    def get_load(self, idx: int):
        """
        Returns instance of load
        :param idx: load ID to return instance
        :return: instance of laad
        """
        return self._loads[idx]

    def get_power_factor(self):
        return self._power_factor

    def set_power_factor(self, pf: float | torch.Tensor):
        self._power_factor = pf

    def get_phase_shift_angle(self) -> float | torch.Tensor:
        return torch.arccos(self._power_factor)

    def set_load_voltage(self, load_idx: int, voltage: float | torch.Tensor) -> None:
        """
        Update node voltage

        :param load_idx: ID of node to update voltage level
        :param voltage: Node voltage [Volt]
        :return: None
        """
        self._loads[load_idx].set_voltage(voltage)

    def get_load_voltage(self, load_idx: int) -> float | torch.Tensor:
        """
        Get node voltage of node_idx
        :param load_idx: ID of Node to return voltage
        :return: voltage [Volts]
        """
        return self._loads[load_idx].get_voltage()

    def get_load_current(self, load_idx: int) -> float | torch.Tensor:
        """
        Return load current of specified load ID
        :param load_idx: Load ID
        :return: Current in Amps
        """
        return self._loads[load_idx].get_current()

    def get_voltage(self) -> float | torch.Tensor:
        """
        Nominal network voltage. Will be used as default node voltage.
        Call compute_partial_voltages() method to compute correct node voltage.

        :return: Network voltage [Volt]
        """
        return self._voltage

    def set_voltage(self, voltage: float | torch.Tensor) -> None:
        """
        Set nominal Network voltage. Will be used as default node voltage.
        :param voltage: Network voltage [Volt]
        :return: None
        """
        self._voltage = voltage

    def get_current(self) -> float | torch.Tensor:
        """
        Calculates sum of all load currents
        Note: execute compute_partial_voltages() before calling this method!

        :return: Current [Amps]
        """
        return sum(load.get_current() for load in self._loads)

    def get_current_at_load(self, idx: int):
        return sum(load.get_current() for load in self._loads[idx:])

    def get_dU_percent(self, load_idx: int = None) -> float | torch.Tensor:
        """
        Get voltage drop (dU) in percent on actual line
        :return: percent voltage drop
        """
        dU = self.get_dUx(load_idx)
        return (dU / self._voltage) * 100

    def get_load_distance(self, idx: int) -> float | torch.Tensor:
        distance = sum([self._loads[i].get_length() for i in range(idx + 1)])
        return distance

    def get_line_length(self) -> float | torch.Tensor:
        return self.get_load_distance(self.get_load_count() - 1)

    def get_dUx(self, load_idx: int = None) -> float | torch.Tensor:
        """
        Calculate delta U at node_id.

        Note: execute method compute_partial_voltages() before executing thi method in order to increase precision.

        :param load_idx: Node ID of load partial voltage to calculate. 0=first node, None=last node
        :return: delta U at node_id
        """

        if load_idx is None:
            load_idx = self.get_load_count()
        else:
            load_idx += 1

        current_list = [self._loads[i].get_current() for i in range(load_idx)]
        current_list = torch.stack(current_list)

        assert len(current_list), "No currents for calculation available"

        if load_idx is not None:
            current_list[-1] += sum(load.get_current() for load in self._loads[load_idx:])

        length = 0
        Mw = 0
        Mb = 0

        for load, current in zip(self._loads[:load_idx], current_list):
            length += load.get_length()

            # calc current momentum
            Mw += current * length * load.get_power_factor()

            # calc current reactive moment
            Mb += current * length * torch.tan(load.get_phase_shift_angle())

        # conductor_impedance = np.vectorize(complex)(self._conductor_resistivity, self._conductor_reactance)
        # current_moment = np.vectorize(complex)(Mw, -Mb)

        conductor_impedance = (self._resistivity / 1000) + 1j * self._reactance
        current_moment = Mw - 1j * Mb

        dUx = abs(conductor_impedance[:current_moment.size(0)] * current_moment)

        # return torch.abs(conductor_impedance[:current_moment.size] * current_moment)

        return dUx

    def compute_partial_voltages(self, iterations: int = 2) -> None:
        """
        Computes all partial voltages by iterating over the nodes multiple times (param: iteration).
        Note: In case of nested line increase iterations count
        :param iterations: iterations for node partial voltages optimization
        :return: None
        """

        # recompute partial voltages
        for i in range(iterations):
            for x in range(self.get_load_count()):  # iterate over all nodes
                drop_voltage = self.get_voltage() - self.get_dUx(x)  # get partial voltage at specific node
                self._loads[x].set_voltage(drop_voltage.detach())
