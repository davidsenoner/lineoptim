import json
import logging
import torch
from torch import nn
from torch.optim import Adam
from torch import tensor
import matplotlib.pyplot as plt

from lineoptim.components import Line, compute_partial_voltages

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)-11s - %(levelname)-7s - %(message)s",
)

# disable logging for matplotlib
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# disable logging for PIL
logging.getLogger('PIL').setLevel(logging.WARNING)


class NetworkOptimizer(nn.Module):
    def __init__(self, line, **kwargs):
        super(NetworkOptimizer, self).__init__()

        self.line = line
        self.v_nominal = line['v_nominal']

        # optimisation params
        self.resistivity = nn.Parameter(line.get_resistivity_tensor(), requires_grad=True)

        print(f"Initial resistivity: {self.resistivity}")

    def forward(self):

        self.line.set_resistivity_tensor(self.resistivity)  # update resistivity

        compute_partial_voltages(self.line, iterations=5)

        dux = self.line.get_residual_voltage_tensor()  # compute residual voltage

        voltage_drop_percent = (1 - (dux / self.line['v_nominal'])) * 100  # convert to %

        return voltage_drop_percent


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

    def optimize(self, line_idx=0, epochs: int = 200, lr: float = 0.01, max_v_drop: float = 5.0, auto_stop=True) -> None:
        """
        Optimize network
        :param line_idx: Line ID to optimize
        :param epochs: number of epochs
        :param lr: learning rate
        :param max_v_drop: max voltage drop in % on all lines
        :param auto_stop: Stops optimization if loss is less than 1e-3
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

        target_data = torch.full_like(resistivity_t, fill_value=max_v_drop)  # compute loss on voltage drop in %
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

            if auto_stop and loss.item() < 1e-3:
                break

        # print results
        print(f'Predictions: {predictions}')
        print(f'Resulting resistivity: {model.resistivity}')

        # plot the loss
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
