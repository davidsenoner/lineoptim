import json
import torch
from torch import nn
from torch.optim import Adam
from torch import tensor
import matplotlib.pyplot as plt

from components.motor import Motor
from components.line import Line


class LineOptimizer(nn.Module):
    def __init__(self, config, **kwargs):
        super(LineOptimizer, self).__init__()

        self.ue_voltage = tensor(config["ue_voltage"])  # get main line voltage

        self.lines = []

        # create line instance from config/json
        self.main_line = self.create_line(config)
        self.main_line.compute_partial_voltages()

        # start resistance for optimizer
        start_resistivity = 0.125

        shape = (self.line_num, self.cores_num)
        params = torch.full(fill_value=start_resistivity, size=shape, dtype=torch.float)  # resistances to optim

        # optimisation params
        self.resistivity = nn.Parameter(params, requires_grad=True)

    @property
    def line_num(self) -> int:
        return len(self.lines)

    @property
    def cores_num(self) -> int:
        return len(self.ue_voltage)

    def forward(self):
        resistivity = self.resistivity
        lines = self.lines

        # set resistivity for each line
        for line, res in zip(lines, resistivity):
            line.set_resistivity(res)

        # compute partial voltages for each line
        for _ in range(5):
            for line in reversed(lines):
                line.compute_partial_voltages()  # TODO: define qnty of iterations in case of nested lines (see default)

        voltage_drop = []

        # calculate voltage drop for each line
        for line in lines:
            dux = line.get_voltage() - line.get_dUx()  # calc voltage drop on actual line
            voltage_drop_percent = (1 - (dux / self.ue_voltage)) * 100  # convert to %
            voltage_drop.append(voltage_drop_percent)

        return torch.stack(voltage_drop, dim=0)

    def create_line(self, net, level=0) -> Line:
        ln = Line(**net, level=level)
        for comp in net["components"]:
            if comp["type"] == "Motor":
                ln.add(load=Motor(**comp))
            if comp["type"] == "Line":
                ln.add(load=self.create_line(comp, level=(level + 1)))

        self.lines.append(ln)

        return ln


NUM_EPOCHS = 200  # number of epochs
NETWORK_CONFIG = 'resources/line_structure.json'  # network configuration file

if __name__ == '__main__':

    # target voltage drop in %
    target_voltage_drop = 5.0

    # load network configuration
    with open(NETWORK_CONFIG, 'r') as json_file:
        loaded_network = json.load(json_file)
        model = LineOptimizer(loaded_network)

    # shape of target data
    shape = (model.line_num, model.cores_num)

    target_data = torch.full(size=shape, fill_value=target_voltage_drop, dtype=torch.float)  # voltage drop in %
    losses = []

    criterion = nn.MSELoss()  # mean square error
    optimizer = Adam(model.parameters(), lr=0.01)

    for epoch in (range(NUM_EPOCHS)):

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
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {loss.item()}')

    # print results
    print(f'Predictions: {predictions}')

    # plot the loss
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
