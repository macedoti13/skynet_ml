from skynet_ml.nn.initializers.base import BaseInitializer
import numpy as np


class Uniform(BaseInitializer):
    """
    Initializes weights by drawing samples from a uniform distribution.

    The Uniform initializer generates weight values from a uniform distribution within a specified range (low_limit to high_limit).
    It can be useful when you want to ensure that the initialized weights are spread out uniformly across a specific range.

    Attributes:
        low_limit (float): Lower limit of the uniform distribution.
        high_limit (float): Upper limit of the uniform distribution.
        name (str): Name representation for the initializer, useful for debugging and logging.
    """


    def __init__(self, low_limit: float = -1.0, high_limit: float = 1.0) -> None:
        """
        Initializes the uniform weight initializer.

        Args:
            low_limit (float, optional): Lower bound for the uniform distribution. Defaults to -1.0.
            high_limit (float, optional): Upper bound for the uniform distribution. Defaults to 1.0.
        """
        self.low_limit = low_limit
        self.high_limit = high_limit
        self.name = f"uniform_{str(low_limit)}_{str(high_limit)}"
        

    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Initializes the weights matrix by drawing samples from the specified uniform distribution.

        Args:
            input_dim (int): Number of input features or units from the previous layer.
            n_units (int): Number of units in the current layer for which weights need to be initialized.

        Returns:
            np.array: Weight matrix of shape (input_dim, n_units) initialized with values drawn from the specified uniform distribution.
        """
        return np.random.uniform(self.low_limit, self.high_limit, (input_dim, n_units))
