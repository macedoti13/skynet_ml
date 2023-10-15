from skynet_ml.nn.initializers.base import BaseInitializer
import numpy as np


class HeNormal(BaseInitializer):
    """
    Initializes weights using the He normal initialization method.

    The He normal initializer is designed for layers with the ReLU (and variants) activation functions.
    It initializes the weights from a normal distribution with mean 0 and standard deviation sqrt(2 / n), 
    where n is the number of input units in the weight tensor.

    Attributes:
        name (str): Name representation for the initializer, useful for debugging and logging.
    """


    def __init__(self) -> None:
        """
        Initializes the He normal weight initializer.
        """
        self.name = "he_normal"


    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Initializes the weights matrix using the He normal initialization method.

        Args:
            input_dim (int): Number of input features or units from the previous layer.
            n_units (int): Number of units in the current layer for which weights need to be initialized.

        Returns:
            np.array: Weight matrix of shape (input_dim, n_units) initialized with values drawn from the specified normal distribution.
        """
        std = np.sqrt(2.0 / input_dim)
        return np.random.normal(0.0, std, (input_dim, n_units))


class HeUniform(BaseInitializer):
    """
    Initializes weights using the He uniform initialization method.

    The He uniform initializer is designed for layers with the ReLU (and variants) activation functions.
    It initializes the weights from a uniform distribution within the range [-limit, limit], 
    where limit is sqrt(6 / n) and n is the number of input units in the weight tensor.

    Attributes:
        name (str): Name representation for the initializer, useful for debugging and logging.
    """


    def __init__(self) -> None:
        """
        Initializes the He uniform weight initializer.
        """
        self.name = "he_uniform"


    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Initializes the weights matrix using the He uniform initialization method.

        Args:
            input_dim (int): Number of input features or units from the previous layer.
            n_units (int): Number of units in the current layer for which weights need to be initialized.

        Returns:
            np.array: Weight matrix of shape (input_dim, n_units) initialized with values drawn from the specified uniform distribution.
        """
        limit = np.sqrt(6.0 / input_dim)
        return np.random.uniform(-limit, limit, (input_dim, n_units))
