import torch
from collections import OrderedDict

from lineoptim.components import Line


class VoltageAccessor:
    def __init__(self, line: Line, cores: OrderedDict):
        self._line = line
        self._cores = cores

    def __repr__(self):
        return repr(torch.stack([load['v_nominal'] for load in self._line['loads']]))

    @property
    def shape(self):
        return torch.stack([load['v_nominal'] for load in self._line['loads']]).shape

    @property
    def cores(self) -> OrderedDict:
        return self._cores

    @cores.setter
    def cores(self, cores: OrderedDict):
        self._cores = cores

    @property
    def cores_len(self) -> int:
        return len(self._cores)

    def recompute(self, iterations=2):
        """ Recompute partial voltages """
        from lineoptim.components import compute_partial_voltages

        compute_partial_voltages(self._line, iterations=iterations)

    def mean(self):
        """ Mean line voltage """
        val = []
        for core_idx, core in enumerate(self._cores.keys()):
            val.append(torch.mean(torch.stack([load['v_nominal'][core_idx] for load in self._line['loads']]), dim=0))

        return torch.stack(val)

    def min(self):
        """ Minimum line voltage """
        val = []
        for core_idx, core in enumerate(self._cores.keys()):
            val.append(torch.min(torch.stack([load['v_nominal'][core_idx] for load in self._line['loads']]), dim=0).values)

        return torch.stack(val)

    def max(self):
        """ Maximum line voltage """
        val = []
        for core_idx, core in enumerate(self._cores.keys()):
            val.append(torch.max(torch.stack([load['v_nominal'][core_idx] for load in self._line['loads']]), dim=0).values)

        return torch.stack(val)

    def std(self):
        """ Standard deviation of line voltage """
        val = []
        for core_idx, core in enumerate(self._cores.keys()):
            val.append(torch.std(torch.stack([load['v_nominal'][core_idx] for load in self._line['loads']]), dim=0))

        return torch.stack(val)

    def plot(self, ax=None):
        """
        Plot line voltage curve
        Pass fig and ax if you want to plot on existing figure
        """
        import matplotlib.pyplot as plt

        plt.style.use('bmh')
        fig = plt.figure()

        if ax is None:
            ax = fig.add_subplot(111)

        position = [load['position'] for load in self._line['loads']]
        x_ticks = [f'{load["name"]}\n{load["position"]}m' for load in self._line['loads']]

        for core_idx, core in enumerate(self._cores.keys()):
            voltages = torch.stack([load['v_nominal'][core_idx] for load in self._line['loads']])
            ax.plot(position, voltages, marker='o', label=core, color=self._cores[core])

        ax.set_title('Line voltage curve')
        ax.set(xlabel='Loads', ylabel='Voltage (V)')
        ax.set_xticks(position, x_ticks)
        ax.legend(loc='upper right', ncol=3)
        fig.show()

        return fig, ax


class VoltageUnbalanceAccessor:
    def __init__(self, line: Line, cores: OrderedDict):
        self._line = line
        self._cores = cores

    def __repr__(self):
        return repr(self._calc_unbalance())

    @property
    def cores(self) -> OrderedDict:
        return self._cores

    @cores.setter
    def cores(self, cores: OrderedDict):
        self._cores = cores

    @property
    def cores_len(self) -> int:
        return len(self._cores)

    def _calc_unbalance(self):
        """
        Calculate voltage unbalance using NEMA definition
        Note: This equation is not taking phase angles in account.
        TODO: implement IEC definition or True definition
        """
        assert self.cores_len == 3, 'Voltage unbalance is supported only for 3-phase systems'
        voltages = torch.stack([load['v_nominal'] for load in self._line['loads']])
        # mean voltages
        mean = torch.mean(voltages, dim=1)
        # max deviation from mean
        max_deviation = torch.max(torch.abs(voltages - mean.unsqueeze(1)), dim=1).values
        # voltage unbalance
        return max_deviation / mean * 100

    def plot(self, ax=None):
        """ Voltage unbalance """
        assert self.cores_len == 3, 'Voltage unbalance is supported only for 3-phase systems'
        import matplotlib.pyplot as plt

        plt.style.use('bmh')
        fig = plt.figure()

        if ax is None:
            ax = fig.add_subplot(111)

        position = [load['position'] for load in self._line['loads']]
        x_ticks = [f'{load["name"]}\n{load["position"]}m' for load in self._line['loads']]
        unbalance = self._calc_unbalance()

        ax.plot(position, unbalance, marker='o', label='Voltage unbalance', color='black')

        ax.set_title('Line Voltage Unbalance curve')
        ax.set(xlabel='Loads', ylabel='Voltage Unbalance (%)')
        ax.set_xticks(position, x_ticks)
        ax.legend(loc='upper right', ncol=3)
        fig.show()

        return fig, ax