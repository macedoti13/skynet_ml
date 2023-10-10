# import das layers aqui
from skynet_ml.nn.layers.layer import Layer
from abc import ABC, abstractmethod


class Optimizer(ABC):
    """
    Base class for optimization algorithms used in training neural network models.

    Provides a basic structure for updating model parameters using the gradients computed during backpropagation.

    Subclasses should implement the `update` method to specify how the optimizer modifies each layer's weights.

    Attributes:
    -----------
    - learning_rate (float): The learning rate for the optimization algorithm.

    Methods:
    --------
    - update(layer) -> None:
        Abstract method that updates a given layer's weights.
    - step(layers) -> None: 
        Iterates through each layer and applies the update.
    """
    
    
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate
    
        
    @abstractmethod
    def update(self, layer: Layer) -> None:
        """
        Abstract method to update a given layer's weights.

        This method should be implemented by subclasses to specify the logic for updating layer weights.

        Parameters:
        - layer (Layer): The layer whose weights need to be updated.
        """
        pass
    
    
    def step(self, layers: list) -> None:
        """
        Iterates through each layer and applies the update.

        Calls the `update` method on each layer in the provided list.

        Parameters:
        - layers (list[Layer]): List of layers in the model.
        """
        for layer in layers:
            self.update(layer)


    @classmethod
    def get_config(self) -> dict:
        """
        Returns a dictionary containing the configuration of the optimizer.

        Returns:
        - config (dict): Configuration of the optimizer.
        """
        return {}
