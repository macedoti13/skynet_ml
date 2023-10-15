from skynet_ml.nn.initializers.base import BaseInitializer
import numpy as np


class Normal(BaseInitializer):
    """
    Initializes weights by drawing samples from a normal (Gaussian) distribution.

    This initializer generates weight values from a normal distribution with a specified mean and standard deviation.
    It can be useful in certain scenarios where weights need to be initialized with small random values.

    Attributes:
        mean (float): Mean of the normal distribution to draw samples from.
        std (float): Standard deviation of the normal distribution to draw samples from.
        name (str): Name representation for the initializer, useful for debugging and logging.
    """


    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        """
        Initializes the normal weight initializer.

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 1.0.
        """
        self.mean = mean
        self.std = std
        self.name = f"normal_{str(mean)}_{str(std)}"
        

    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Initializes the weights matrix by drawing samples from the specified normal distribution.

        Args:
            input_dim (int): Number of input features or units from the previous layer.
            n_units (int): Number of units in the current layer for which weights need to be initialized.

        Returns:
            np.array: Weight matrix of shape (input_dim, n_units) initialized with values drawn from the specified normal distribution.
        """
        return np.random.normal(self.mean, self.std, (input_dim, n_units))
