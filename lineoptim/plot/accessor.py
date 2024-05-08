import torch
from collections import OrderedDict

from lineoptim.components import Line
from lineoptim.components import get_current


class BaseAccessor:
    def __init__(self, line: Line, cores: OrderedDict):
        self._line = line
        self._cores = cores

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


class VoltageAccessor(BaseAccessor):
    def __init__(self, line: Line, cores: OrderedDict):
        super().__init__(line, cores)

    def __repr__(self):
        return repr(torch.stack([load['v_nominal'] for load in self._line['loads']]))

    def __getitem__(self, index):
        sel = self._line['loads'][index]
        if isinstance(sel, dict):
            return sel['v_nominal']
        elif isinstance(sel, list):
            return torch.stack([load['v_nominal'] for load in sel])
        else:
            raise ValueError('Invalid index')

    @property
    def shape(self):
        return torch.stack([load['v_nominal'] for load in self._line['loads']]).shape

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
            val.append(
                torch.min(torch.stack([load['v_nominal'][core_idx] for load in self._line['loads']]), dim=0).values)

        return torch.stack(val)

    def max(self):
        """ Maximum line voltage """
        val = []
        for core_idx, core in enumerate(self._cores.keys()):
            val.append(
                torch.max(torch.stack([load['v_nominal'][core_idx] for load in self._line['loads']]), dim=0).values)

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


class SumCurrentAccessor(BaseAccessor):
    def __init__(self, line: Line, cores: OrderedDict):
        super().__init__(line, cores)

    def __repr__(self):
        return repr(torch.stack([self._line.get_spot_current(idx) for idx in range(len(self._line['loads']))]))

    def __getitem__(self, index):
        sel = self._line['loads'][index]
        if isinstance(sel, dict):
            return self._line.get_spot_current(index)
        elif isinstance(sel, list):
            return torch.stack([self._line.get_spot_current(idx) for idx in range(len(sel))])
        else:
            raise ValueError('Invalid index')

    def min(self):
        return torch.min(torch.stack([self._line.get_spot_current(idx) for idx in range(len(self._line['loads']))]))

    def max(self):
        return torch.max(torch.stack([self._line.get_spot_current(idx) for idx in range(len(self._line['loads']))]))

    def mean(self):
        return torch.mean(torch.stack([self._line.get_spot_current(idx) for idx in range(len(self._line['loads']))]))

    def std(self):
        return torch.std(torch.stack([self._line.get_spot_current(idx) for idx in range(len(self._line['loads']))]))

    def plot(self, ax=None):
        """
        Plot spot current curve
        Pass fig and ax if you want to plot on existing figure
        """
        import matplotlib.pyplot as plt

        plt.style.use('bmh')
        fig = plt.figure()

        if ax is None:
            ax = fig.add_subplot(111)

        position = [load['position'] for load in self._line['loads']]
        x_ticks = [f'{load["name"]}\n{load["position"]}m' for load in self._line['loads']
                   if 'position' in load.keys()]

        spot_current = torch.stack([self._line.get_spot_current(idx) for idx in range(len(self._line['loads']))])

        for core_idx, core in enumerate(self._cores.keys()):
            ax.plot(position, spot_current[:, core_idx], marker='o', label=core, color=self._cores[core])

        ax.set_title('Spot current curve')
        ax.set(xlabel='Loads', ylabel='Current (A)')
        ax.set_xticks(position, x_ticks)
        ax.legend(loc='upper right', ncol=3)
        fig.show()

        return fig, ax


class CurrentAccessor(BaseAccessor):
    def __init__(self, line: Line, cores: OrderedDict):
        super().__init__(line, cores)

    def __repr__(self):
        return repr(torch.stack([get_current(load) for load in self._line['loads']]))

    def __getitem__(self, index):
        sel = self._line['loads'][index]
        if isinstance(sel, dict):
            return get_current(sel)
        elif isinstance(sel, list):
            return torch.stack([get_current(load) for load in sel])
        else:
            raise ValueError('Invalid index')

    def min(self):
        return torch.min(torch.stack([get_current(load) for load in self._line['loads']]))

    def max(self):
        return torch.max(torch.stack([get_current(load) for load in self._line['loads']]))

    def mean(self):
        return torch.mean(torch.stack([get_current(load) for load in self._line['loads']]))

    def std(self):
        return torch.std(torch.stack([get_current(load) for load in self._line['loads']]))

    def plot(self, ax=None):
        """
        Plot spot current curve
        Pass fig and ax if you want to plot on existing figure
        """
        import matplotlib.pyplot as plt

        plt.style.use('bmh')
        fig = plt.figure()

        if ax is None:
            ax = fig.add_subplot(111)

        position = [load['position'] for load in self._line['loads']]
        x_ticks = [f'{load["name"]}\n{load["position"]}m' for load in self._line['loads']
                   if 'position' in load.keys()]

        current = torch.stack([get_current(load) for load in self._line['loads']])

        for core_idx, core in enumerate(self._cores.keys()):
            ax.plot(position, current[:, core_idx], marker='o', label=core, color=self._cores[core])

        ax.set_title('Load current curve')
        ax.set(xlabel='Loads', ylabel='Current (A)')
        ax.set_xticks(position, x_ticks)
        ax.legend(loc='upper right', ncol=3)
        fig.show()

        return fig, ax


class VoltageUnbalanceAccessor(BaseAccessor):
    def __init__(self, line: Line, cores: OrderedDict):
        super().__init__(line, cores)

    def __repr__(self):
        return repr(self._calc_unbalance())

    def __getitem__(self, index):
        sel = self._line['loads'][index]
        if isinstance(sel, dict):
            return self._calc_unbalance()[index]
        elif isinstance(sel, list):
            return torch.stack([self._calc_unbalance()[idx] for idx in range(len(sel))])
        else:
            raise ValueError('Invalid index')

    def min(self):
        return torch.min(self._calc_unbalance())

    def max(self):
        return torch.max(self._calc_unbalance())

    def mean(self):
        return torch.mean(self._calc_unbalance())

    def std(self):
        return torch.std(self._calc_unbalance())

    def _calc_unbalance(self):
        """
        Calculate voltage unbalance using NEMA definition
        Note: This equation is not taking phase angles into account.
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

        ax.set_title('Load Voltage Unbalance curve')
        ax.set(xlabel='Loads', ylabel='Voltage Unbalance (%)')
        ax.set_xticks(position, x_ticks)
        ax.legend(loc='upper right', ncol=3)
        fig.show()

        return fig, ax


class CurrentUnbalanceAccessor(BaseAccessor):
    def __init__(self, line: Line, cores: OrderedDict):
        super().__init__(line, cores)

    def __repr__(self):
        return repr(self._calc_unbalance())

    def __getitem__(self, index):
        sel = self._line['loads'][index]
        if isinstance(sel, dict):
            return self._calc_unbalance()[index]
        elif isinstance(sel, list):
            return torch.stack([self._calc_unbalance()[idx] for idx in range(len(sel))])
        else:
            raise ValueError('Invalid index')

    def min(self):
        return torch.min(self._calc_unbalance())

    def max(self):
        return torch.max(self._calc_unbalance())

    def mean(self):
        return torch.mean(self._calc_unbalance())

    def std(self):
        return torch.std(self._calc_unbalance())

    def _calc_unbalance(self):
        """
        Calculate current unbalance using NEMA definition
        Note: This equation is not taking phase angles into account.
        TODO: implement IEC definition or True definition
                """
        assert self.cores_len == 3, 'Current unbalance is supported only for 3-phase systems'
        currents = torch.stack([get_current(load) for load in self._line['loads']])
        # mean voltages
        mean = torch.mean(currents, dim=1)
        # max deviation from mean
        max_deviation = torch.max(torch.abs(currents - mean.unsqueeze(1)), dim=1).values
        # voltage unbalance
        return max_deviation / mean * 100

    def plot(self, ax=None):
        """ Current unbalance """
        assert self.cores_len == 3, 'Current unbalance is supported only for 3-phase systems'
        import matplotlib.pyplot as plt

        plt.style.use('bmh')
        fig = plt.figure()

        if ax is None:
            ax = fig.add_subplot(111)

        position = [load['position'] for load in self._line['loads']]
        x_ticks = [f'{load["name"]}\n{load["position"]}m' for load in self._line['loads']
                   if 'position' in load.keys()]
        unbalance = self._calc_unbalance()

        ax.plot(position, unbalance, marker='o', label='Current unbalance', color='black')

        ax.set_title('Load Current Unbalance curve')
        ax.set(xlabel='Loads', ylabel='Current Unbalance (%)')
        ax.set_xticks(position, x_ticks)
        ax.legend(loc='upper right', ncol=3)
        fig.show()

        return fig, ax
