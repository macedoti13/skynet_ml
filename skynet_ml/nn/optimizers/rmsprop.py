from skynet_ml.nn.optimizers.base import BaseOptimizer
from skynet_ml.nn.layers.base import BaseLayer
import numpy as np


class RMSProp(BaseOptimizer):
    """
    Root Mean Square Propagation (RMSProp) optimizer.
    
    RMSProp is an adaptive learning rate optimization algorithm designed to address Adagrad's 
    radically diminishing learning rates. It utilizes a moving average of squared gradients 
    to normalize the gradient.
    """    
    
    
    def __init__(self, learning_rate: float = 0.01, beta: float = 0.9) -> None:
        """
        Initializes the RMSProp optimizer with the given learning rate and decay factor.

        Args:
            learning_rate (float, optional): Learning rate used in the parameter updates. Defaults to 0.01.
            beta (float, optional): Decay factor for the moving average of squared gradients. 
                                    A value close to 1.0 will have a longer memory of past gradients, 
                                    and a value close to 0.0 will only consider recent gradients. Defaults to 0.9.
        """        
        super().__init__(learning_rate)
        self.beta = beta
        self.name = f"rmsprop_{str(self.learning_rate)}_{str(self.beta)}"
            
            
    def update(self, layer: BaseLayer) -> None:
        """
        Performs a parameter update using the RMSProp algorithm.

        Adjusts the weights and biases (if present) of the given layer based on the moving average of squared gradients
        to ensure more robust convergence.

        Args:
            layer (BaseLayer): The neural network layer whose parameters (weights and biases) 
            need to be updated based on the moving average of squared gradients.
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
        Updates the moving average of squared gradients for the given layer.

        Computes the moving average of the squared gradients of the weights and biases (if present) of the 
        layer to be used in the subsequent update steps.

        Args:
            layer (BaseLayer): The neural network layer for which the moving average of squared 
            gradients is being updated.
        """

        # initializing the v_weights and v_bias attributes if they don't exist
        if not hasattr(layer, 'v_weights'):
            layer.initialize_velocity() 
            
        # updating the moving average of squared gradients for weights
        layer.v_weights = self.beta * layer.v_weights + (1 - self.beta) * layer.d_weights ** 2 
        
        # updating the moving average of squared gradients for bias
        if layer.has_bias:
            layer.v_bias = self.beta * layer.v_bias + (1 - self.beta) * layer.d_bias ** 2 
