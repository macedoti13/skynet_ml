from skynet_ml.nn.optimizers.base import BaseOptimizer
from skynet_ml.nn.layers.base import BaseLayer
import numpy as np


class AdaGrad(BaseOptimizer):
    """
    Adaptive Gradient Algorithm (AdaGrad) optimizer.
    
    AdaGrad is an adaptive learning rate optimization algorithm designed to improve 
    convergence and training stability by scaling learning rates with respect to 
    the historical gradient at each parameter.

    Args:
        BaseOptimizer (BaseOptimizer): Inherits basic optimizer properties and methods.
    """    
    
    def __init__(self, learning_rate: float = 0.01) -> None:
        """
        Initializes the RMSProp optimizer with the given learning rate and decay factor.

        Args:
            learning_rate (float, optional): Learning rate used in the parameter updates. Defaults to 0.01.
        """        
        super().__init__(learning_rate)
        self.name = f"adagrad_{str(self.learning_rate)}"
    
    
    def update(self, layer: BaseLayer) -> None:
        """
        Performs a parameter update using the AdaGrad algorithm.

        Adjusts the weights and biases (if present) of the given layer based on the 
        historical gradient to ensure more robust convergence.

        Args:
            layer (BaseLayer): The neural network layer whose parameters (weights and biases) 
            need to be updated based on the accumulated squared gradients.
        """
        
        epsilon = 1e-15
        self._update_v(layer)
        
        # normalizing the gradient for weights
        gradient_normalization_weights = np.sqrt(layer.v_weights) + epsilon 
        # normalizing the learning rate
        normalized_learning_rate = self.learning_rate / gradient_normalization_weights 
        # updating the weights
        layer.weights -= normalized_learning_rate * layer.d_weights 
        
        
        if layer.has_bias:    
            # normalizing the gradient for bias
            gradient_normalization_bias = np.sqrt(layer.v_bias) + epsilon 
            # normalizing the learning rate
            normalized_learning_rate = self.learning_rate / gradient_normalization_bias 
            # updating the bias
            layer.bias -= normalized_learning_rate * layer.d_bias 
            
            
    def _update_v(self, layer: BaseLayer) -> None:
        """
        Updates the accumulated squared gradients for the given layer.

        Accumulates the squared gradients of the weights and biases (if present) of the 
        layer to be used in the subsequent update steps.

        Args:
            layer (BaseLayer): The neural network layer for which the accumulated squared 
            gradients are being updated.
        """
        
        # initializing the v_weights and v_bias attributes if they don't exist
        if not hasattr(layer, 'v_weights'):
            layer.initialize_velocity()
            
        # updating the v_weights
        layer.v_weights += layer.d_weights ** 2 
        
        # updating the v_bias
        if layer.has_bias:
            layer.v_bias += layer.d_bias ** 2