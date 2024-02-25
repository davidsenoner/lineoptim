import numpy as np
from numpy.typing import NDArray
import torch
from torch import tensor, Tensor


class Motor:
    """
    Asynchronous motor with 3phases (no neutral conductor)
    """

    def __init__(self, active_power: float,
                 voltage: float | NDArray | Tensor = 400,
                 efficiency: float = 1,
                 power_factor: float = 0.9,
                 distance: float = 0,
                 **kwargs
                 ):
        self._voltage = tensor(voltage)
        self._active_power = tensor([active_power])
        self._efficiency = tensor([efficiency])
        self._power_factor = tensor([power_factor])
        self._distance = distance

    def get_length(self) -> float:
        """
        Distance between actual node and precedent node
        :return: Distance in meters
        """
        return self._distance

    def __str__(self):
        return f'Motor'

    def get_voltage(self):
        return self._voltage

    def set_voltage(self, voltage: float) -> None:
        self._voltage = voltage

    def get_active_power(self) -> float:
        return self._active_power

    def set_active_power(self, power: float) -> None:
        self._active_power = power

    def get_efficiency(self):
        return self._efficiency

    def get_mechanical_power(self):
        return self._active_power / self._efficiency

    def get_power_factor(self):
        return self._power_factor

    def set_power_factor(self, pf: float):
        self._power_factor = pf

    def get_phase_shift_angle(self) -> float | Tensor:
        return torch.arccos(self._power_factor)

    def get_current(self):
        return self._active_power / (self._voltage * np.sqrt(3) * self._power_factor)
