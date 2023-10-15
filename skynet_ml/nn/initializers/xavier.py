from skynet_ml.nn.initializers.base import BaseInitializer
import numpy as np


class XavierNormal(BaseInitializer):
    """
    Initializes weights using the Xavier (or Glorot) normal initialization method.

    The Xavier normal initializer is designed for layers with the tanh activation function.
    It initializes the weights from a normal distribution with mean 0 and standard deviation sqrt(1 / n), 
    where n is the number of input units in the weight tensor.

    Attributes:
        name (str): Name representation for the initializer, useful for debugging and logging.
    """


    def __init__(self) -> None:
        """
        Initializes the Xavier normal weight initializer.
        """
        self.name = "xavier_normal"


    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Initializes the weights matrix using the Xavier normal initialization method.

        Args:
            input_dim (int): Number of input features or units from the previous layer.
            n_units (int): Number of units in the current layer for which weights need to be initialized.

        Returns:
            np.array: Weight matrix of shape (input_dim, n_units) initialized with values drawn from the specified normal distribution.
        """
        std = np.sqrt(1.0 / input_dim)
        return np.random.normal(0.0, std, (input_dim, n_units))


class XavierUniform(BaseInitializer):
    """
    Initializes weights using the Xavier (or Glorot) uniform initialization method.

    The Xavier uniform initializer is designed for layers with the tanh activation function.
    It initializes the weights from a uniform distribution within the range [-limit, limit], 
    where limit is sqrt(6 / (n + m)) and n is the number of input units and m is the number of output units in the weight tensor.

    Attributes:
        name (str): Name representation for the initializer, useful for debugging and logging.
    """


    def __init__(self) -> None:
        """
        Initializes the Xavier uniform weight initializer.
        """
        self.name = "xavier_uniform"


    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Initializes the weights matrix using the Xavier uniform initialization method.

        Args:
            input_dim (int): Number of input features or units from the previous layer.
            n_units (int): Number of units in the current layer for which weights need to be initialized.

        Returns:
            np.array: Weight matrix of shape (input_dim, n_units) initialized with values drawn from the specified uniform distribution.
        """
        limit = np.sqrt(6.0 / (input_dim + n_units))
        return np.random.uniform(-limit, limit, (input_dim, n_units))
