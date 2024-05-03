import matplotlib.pyplot as plt
import torch

from lineoptim.components import get_current, get_dUx

class PlotData:
    def __init__(self, line, core_names=None):
        """

        :param line: line to plot
        :param core_names: default core names "L1", "L2", "L3", "N", "PE". Set list of names if different
        """
        if core_names is None:
            core_names = ["L1", "L2", "L3", "N", "PE"]
        self.line = line

        self.figure, self.ax = plt.subplots(2, 2)

        self._core_names = core_names
        self.core_style = {
            "L1": "brown",
            "L2": "black",
            "L3": "grey",
            "N": "blue",
            "PE": "green",
        }

    @property
    def line(self):
        return self._line

    @line.setter
    def line(self, value):
        self._line = value

    @property
    def core_names(self):
        return self._core_names

    @core_names.setter
    def core_names(self, value):
        self._core_names = value

    @property
    def min_current(self):
        current = [get_current(load) for load in self.line['loads']]
        return min([min(curr) for curr in current]).item()

    @property
    def max_current(self):
        current = [get_current(load) for load in self.line['loads']]
        return max([max(curr) for curr in current]).item()

    @property
    def min_apparent_power(self):
        apparent_power = [load['apparent_power'] for load in self.line['loads']]
        return min(apparent_power)

    @property
    def max_apparent_power(self):
        apparent_power = [load['apparent_power'] for load in self.line['loads']]
        return max(apparent_power)

    @property
    def min_active_power(self):
        active_power = [load['active_power'] for load in self.line['loads']]
        return min(active_power)

    @property
    def max_active_power(self):
        active_power = [load['active_power'] for load in self.line['loads']]
        return max(active_power)

    def plot_line_voltage(self, ax=None, **kwargs):
        """
        Plot line voltage curve
        Pass fig and ax if you want to plot on existing figure
        """

        plt.style.use('bmh')
        fig = plt.figure()

        if ax is None:
            ax = fig.add_subplot(111)

        core_names = kwargs.get("core_names", self._core_names)

        position = [load['position'] for load in self.line['loads']]
        voltage = torch.stack([load['v_nominal'] for load in self.line['loads']])

        x_ticks = [f'{load["name"]}\n{load["position"]}m' for load in self.line['loads']]

        for core, core_name in zip(range(self.line.cores_to_optimize()), core_names):
            ax.plot(position, voltage[:, core], marker='o', label=core_name, color=self.core_style[core_name])

        ax.set_title('Line voltage curve')
        ax.set(xlabel='Loads', ylabel='Voltage (V)')
        ax.set_xticks(position, x_ticks)
        ax.legend(loc='upper right', ncol=3)
        fig.show()

    def plot_apparent_power(self, ax=None, **kwargs):
        """
        Plot apparent power curve
        Pass fig and ax if you want to plot on existing figure
        """

        plt.style.use('bmh')
        fig = plt.figure()

        if ax is None:
            ax = fig.add_subplot(111)

        position = [load['position'] for load in self.line['loads']]
        apparent_power = [load['apparent_power'] for load in self.line['loads']]
        power_factor = [load['power_factor'] for load in self.line['loads']]

        height = (self.max_apparent_power - self.min_apparent_power) // 5 + 1
        x_ticks = [f'{load["name"]}\n{load["position"]}m' for load in self.line['loads']]

        ax.stem(position, apparent_power, basefmt=" ", linefmt="-", markerfmt="o", label='Apparent power (VA)')

        for pos, power, pf in zip(position, apparent_power, power_factor):
            ax.text(pos, power, f'{power:.2f} VA\n{pf=}', fontsize=8, color='black', ha='left', va='bottom', rotation=45)

        ax.set_title('Apparent power curve')
        ax.set(xlabel='Loads', ylabel='Apparent power (VA)')
        ax.set_ylim(max(0, self.min_apparent_power-height), self.max_apparent_power+height)
        ax.set_xticks(position, x_ticks)
        fig.show()

        return fig, ax

    def plot_active_power(self, ax=None, **kwargs):
        """
        Plot active power curve
        Pass fig and ax if you want to plot on existing figure
        """

        plt.style.use('bmh')
        fig = plt.figure()

        if ax is None:
            ax = fig.add_subplot(111)

        position = [load['position'] for load in self.line['loads']]
        active_power = [load['active_power'] for load in self.line['loads']]
        power_factor = [load['power_factor'] for load in self.line['loads']]

        height = (self.max_active_power - self.min_active_power) // 5 + 1
        x_ticks = [f'{load["name"]}\n{load["position"]}m' for load in self.line['loads']]

        ax.stem(position, active_power, basefmt=" ", linefmt="-", markerfmt="o", label='Active power (W)')

        for pos, power, pf in zip(position, active_power, power_factor):
            ax.text(pos, power, f'{power:.2f} W\n{pf=}', fontsize=8, color='black', ha='left', va='bottom', rotation=45)

        ax.set_title('Active power curve')
        ax.set(xlabel='Loads', ylabel='Active power (W)')
        ax.set_ylim(max(0, self.min_active_power-height), self.max_active_power+height)
        ax.set_xticks(position, x_ticks)
        fig.show()

        return fig, ax

    def plot_current(self, ax=None, **kwargs):
        """
        Plot current curve
        Pass fig and ax if you want to plot on existing figure
        """

        plt.style.use('bmh')
        fig = plt.figure()

        if ax is None:
            ax = fig.add_subplot(111)

        position = [load['position'] for load in self.line['loads']]
        current = [get_current(load) for load in self.line['loads']]
        core_names = kwargs.get("core_names", self._core_names)

        x_ticks = [f'{load["name"]}\n{load["position"]}m' for load in self.line['loads']]

        width = (max(position) - min(position)) / 50
        height = (self.max_current - self.min_current) // 5 + 1

        for core, core_name in zip(range(self.line.cores_to_optimize()), core_names):
            offset = (core-1) * width
            for pos, curr in zip(position, current):
                ax.bar(pos+offset, curr[core], width=width, color=self.core_style[core_name])
                ax.text(pos+offset, curr[core], f'  {core_name}={curr[core]:.2f} A', fontsize=8, color='black', ha='left', va='bottom', rotation=90)

        ax.set_title('Current curve')
        ax.set(xlabel='Loads', ylabel='Current (A)')
        ax.set_ylim(max(0, self.min_current-height), self.max_current+height)
        ax.set_xticks(position, x_ticks)
        fig.show()

        return fig, ax