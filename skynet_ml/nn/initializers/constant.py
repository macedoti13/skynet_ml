from skynet_ml.nn.initializers.base import BaseInitializer
import numpy as np


class Constant(BaseInitializer):
    """
    Initializes weights with a constant value.

    This initializer sets all weight values to a given constant. While it can be useful in certain specific
    scenarios, initializing all weights to the same constant can be problematic in neural networks as it 
    leads to all neurons in a layer behaving identically, which can hinder learning.

    Attributes:
        constant (float): The value used to initialize weights.
        name (str): Name representation for the initializer, useful for debugging and logging.
    """


    def __init__(self, constant: float = 1.0) -> None:
        """
        Initializes the constant weight initializer.

        Args:
            constant (float, optional): The value to initialize weights with. Defaults to 1.0.
        """
        self.constant = constant
        self.name = f"constant_{str(constant)}"


    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Initializes the weights matrix with the specified constant value.

        Args:
            input_dim (int): Number of input features or units from the previous layer.
            n_units (int): Number of units in the current layer for which weights need to be initialized.

        Returns:
            np.array: Weight matrix of shape (input_dim, n_units) initialized with the specified constant value.
        """
        return np.full((input_dim, n_units), self.constant)
