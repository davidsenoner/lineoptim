import json
import logging
import torch
from torch import nn
from torch.optim import Adam
from torch import tensor
import matplotlib.pyplot as plt

from lineoptim.components import Line

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)-11s - %(levelname)-7s - %(message)s",
)


class NetworkOptimizer(nn.Module):
    def __init__(self, line, **kwargs):
        super(NetworkOptimizer, self).__init__()

        # optimisation params
        self.resistivity = nn.Parameter(line.get_resistivity_tensor(), requires_grad=True)

    def forward(self):
        resistivity = self.resistivity
        lines = self.lines_to_optimize

        # set resistivity for each line
        for line, res in zip(lines, resistivity):
            line.resistivity = res

        # compute partial voltages for each line
        for _ in range(5):
            for line in reversed(lines):
                line.compute_partial_voltages()  # TODO: define qnty of iterations in case of nested lines (see default)

        voltage_drop = []

        # calculate voltage drop for each line
        for line in lines:
            dux = line.v_nominal - line.get_dUx()  # calc voltage drop on actual line
            voltage_drop_percent = (1 - (dux / self._v_nominal)) * 100  # convert to %
            voltage_drop.append(voltage_drop_percent)

        return torch.stack(voltage_drop, dim=0)


class Network:
    def __init__(self, lines=None):
        if lines is None:
            lines = []
        self._lines = lines

    def add(self, component):

        if isinstance(component, Line):
            self._lines.append(component)
        else:
            raise ValueError(f'Component type is not supported')

    def optimize(self, line_idx=0, epochs: int = 200, lr: float = 0.1, max_v_drop: float = 5.0) -> None:
        """
        Optimize network
        :param line_idx: Line ID to optimize
        :param epochs: number of epochs
        :param lr: learning rate
        :param max_v_drop: max voltage drop in % on all lines
        :return:
        """

        # logging
        logger.info(f'Optimizing network with {epochs} epochs, learning rate {lr} and max voltage drop {max_v_drop}')

        # epochs min 1
        if epochs < 1:
            raise ValueError('Epochs must be greater than 0')

        line = self._lines[line_idx]
        resistivity_t = line.get_resistivity_tensor()

        model = NetworkOptimizer(line)  # create model

        target_data = torch.full_like(resistivity_t, fill_value=max_v_drop)  # voltage drop in %
        losses = []

        criterion = nn.MSELoss()  # mean square error
        optimizer = Adam(model.parameters(), lr=lr)

        for epoch in (range(epochs)):

            # Forward pass
            predictions = model()

            # Calculate the mean square error
            loss = criterion(predictions, target_data)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save the loss
            losses.append(loss.item())

            # Output of progress
            if (epoch + 1) % (epochs / 10) == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

        # print results
        print(f'Predictions: {predictions}')

        # plot the loss
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
