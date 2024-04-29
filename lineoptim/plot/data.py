import matplotlib.pyplot as plt
import torch

from lineoptim.components import get_current, get_dUx


class PlotData:
    def __init__(self, line):
        self.line = line

    def plot(self, **kwargs):
        fig, ax = plt.subplots(2, 2)

        core_prefix = kwargs.get("core_prefix", "L")

        position = [load['position'] for load in self.line['loads']]
        load_current = torch.stack([get_current(load) for load in self.line['loads']])
        active_power = [load['active_power'] for load in self.line['loads']]
        spot_current = torch.stack([self.line.get_spot_current(idx) for idx, load in enumerate(self.line['loads'])])
        partial_voltage = torch.stack([get_dUx(**self.line.dict(), node_id=idx) for idx, load in enumerate(self.line['loads'])])

        for core in range(self.line.cores_to_optimize()):
            # plot load current
            ax[0, 0].plot(position, load_current[:, core], marker='o', label=f'{core_prefix}{core + 1}')
            # plot load active power
            ax[0, 1].plot(position, active_power, marker='o')
            # plot spot current
            ax[1, 0].plot(position, spot_current[:, core], marker='o', label=f'{core_prefix}{core + 1}')
            # plot residual voltage
            ax[1, 1].plot(position, partial_voltage[:, core], marker='o', label=f'{core_prefix}{core + 1}')

        ax[0, 0].set(xlabel='Position (m)', ylabel='Current (A)', title='Load current')
        ax[0, 0].legend()
        ax[0, 0].grid()
        ax[0, 1].set(xlabel='Position (m)', ylabel='Active Power (W)', title='Load active power')
        ax[0, 1].grid()
        ax[1, 0].set(xlabel='Position (m)', ylabel='Spot Current (A)', title='Spot current')
        ax[1, 0].legend()
        ax[1, 0].grid()
        ax[1, 1].set(xlabel='Position (m)', ylabel='Residual Voltage (V)', title='Partial voltage')
        ax[1, 1].legend()
        ax[1, 1].grid()
        fig.show()
