from typing import List, Union, Optional
import numpy as np
import logging
import torch
from torch import nn
from torch.optim import Adam
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
    """
    Neural network optimizer for electrical network parameters.
    
    This class uses PyTorch to optimize conductor resistivity parameters
    to achieve desired voltage drop characteristics.
    """
    
    def __init__(self, line: Line, **kwargs):
        """
        Initialize network optimizer.
        
        Args:
            line: Line instance to optimize
            **kwargs: Additional parameters
        """
        super(NetworkOptimizer, self).__init__()

        self.line = line
        self.v_nominal = line['v_nominal']

        # Optimization parameters
        self.resistivity = nn.Parameter(line.get_resistivity_tensor(), requires_grad=True)

        print(f"Initial resistivity: {self.resistivity}")

    def forward(self) -> torch.Tensor:
        """
        Forward pass: compute voltage drop percentage.
        
        Returns:
            Voltage drop percentage tensor
        """
        self.line.set_resistivity_tensor(self.resistivity)  # Update resistivity
        self.line.recompute()  # Recompute partial voltages
        
        dux = self.line.get_residual_voltage_tensor()  # Compute residual voltage
        voltage_drop_percent = (1 - (dux / self.line['v_nominal'])) * 100  # Convert to %

        return voltage_drop_percent


class Network:
    """
    Electrical network containing multiple lines for optimization.
    
    This class manages a collection of electrical lines and provides
    optimization capabilities for conductor parameters.
    """
    
    def __init__(self, lines: Optional[Union[Line, List[Line]]] = None):
        """
        Initialize network.
        
        Args:
            lines: Single line or list of lines to add to network
        """
        self._lines = []

        if lines:
            if isinstance(lines, list):
                for line in lines:
                    self.add(line)
            elif isinstance(lines, Line):
                self.add(lines)

    def __repr__(self):
        return repr([line['name'] for line in self._lines])

    def __len__(self):
        return len(self._lines)

    def __getitem__(self, idx):
        return self._lines[idx]

    def __iter__(self):
        return iter(self._lines)

    def add(self, component: Line) -> None:
        """
        Add a line component to the network.
        
        Args:
            component: Line instance to add
            
        Raises:
            ValueError: If component is not a Line instance
        """
        if isinstance(component, Line):
            self._lines.append(component)
        else:
            raise ValueError('Component must be a Line instance')

    def optimize(
        self, 
        line_idx: int = 0, 
        epochs: int = 200, 
        lr: float = 0.01, 
        max_v_drop: float = 5.0,
        auto_stop: bool = True
    ) -> None:
        """
        Optimize network parameters using gradient descent.
        
        Args:
            line_idx: Index of line to optimize
            epochs: Number of training epochs
            lr: Learning rate for optimization
            max_v_drop: Maximum allowed voltage drop percentage
            auto_stop: Stop optimization if loss is below threshold
            
        Raises:
            ValueError: If epochs is less than 1
            IndexError: If line_idx is out of range
        """
        # Validation
        if epochs < 1:
            raise ValueError('Epochs must be greater than 0')
        if line_idx >= len(self._lines):
            raise IndexError(f'Line index {line_idx} out of range')

        # Logging
        logger.info(f'Optimizing network with {epochs} epochs, learning rate {lr} and max voltage drop {max_v_drop}')

        line = self._lines[line_idx]
        resistivity_t = line.get_resistivity_tensor()

        model = NetworkOptimizer(line)  # Create model

        target_data = torch.full_like(resistivity_t, fill_value=max_v_drop)
        losses = []

        criterion = nn.MSELoss()  # Mean square error
        optimizer = Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            # Forward pass
            predictions = model()

            # Calculate loss
            loss = criterion(predictions, target_data)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save loss
            losses.append(loss.item())

            # Progress output
            if (epoch + 1) % (epochs // 10) == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}')

            # Early stopping
            if auto_stop and loss.item() < 1e-3:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        # Print results
        print(f'Predictions: {predictions}')
        print(f'Resulting resistivity: {model.resistivity}')

        # Plot optimization progress
        self._plot_optimization_progress(losses)

    def _plot_optimization_progress(self, losses: List[float]) -> None:
        """
        Plot optimization loss over epochs.
        
        Args:
            losses: List of loss values per epoch
        """
        fig, ax = plt.subplots()
        fig.suptitle('Optimization Progress')
        ax.plot(losses)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True)
        fig.show()
