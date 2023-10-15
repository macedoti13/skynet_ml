from skynet_ml.nn.layers.base import BaseLayer
from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    """
    Base class for optimizers used in training neural networks.

    Provides a general structure for updating parameters of neural network layers 
    based on the computed gradients during backpropagation. Specific optimization 
    algorithms should inherit from this class and implement the `update` method.
    """    

    def __init__(self, learning_rate: float = 0.01) -> None:
        """
        Initializes the optimizer with a specified learning rate.

        Args:
            learning_rate (float, optional): The step size used to update the parameters 
            during training. Defaults to 0.01.
        """        
        self.learning_rate = learning_rate
    
    
    @abstractmethod
    def update(self, layer: BaseLayer) -> None:
        """
        Performs a single optimization step for the provided layer.

        This method should be overridden by specific optimizer subclasses, to 
        define the unique updating strategy used by the optimization algorithm.

        Args:
            layer (BaseLayer): The neural network layer whose parameters need to 
            be updated.
        """
        pass
    
    
    def step(self, layers: list) -> None:
        """
        Iteratively updates parameters of a list of neural network layers.

        For each layer in the list, this method calls the `update` method to adjust 
        the layer's parameters based on the optimization strategy.

        Args:
            layers (list): A list containing the neural network layers to be updated.
        """        
        for layer in layers:
            self.update(layer)
