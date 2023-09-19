from skynet_ml.nn.optimizers.optimizer import Optimizer
from skynet_ml.nn.layers.layer import Layer
import numpy as np

class RMSProp(Optimizer):
    """
    Root Mean Square Propagation (RMSProp) optimization algorithm.

    RMSProp is an adaptive learning rate optimization algorithm, which adjusts the learning rate of each parameter
    based on the moving average of the squared gradients up to the current step. It helps to resolve the issues 
    of Adagrad's aggressively decreasing learning rates.

    Attributes
    ----------
    beta : float
        Exponential decay rate for the moving average of squared gradients.
    
    Methods
    -------
    update(layer: Layer) -> None:
        Updates the weights and biases of the provided layer using the RMSProp algorithm.
    
    _update_v(layer: Layer) -> None:
        Internal helper method to update the moving average of squared gradients for a layer.
    """
    
    def __init__(self, learning_rate: float = 0.01, beta: float = 0.9) -> None:
        """
        Initialize the RMSProp optimizer with the given learning rate and beta value.

        Parameters
        ----------
        learning_rate : float, optional
            Step size to adjust the weights during training, by default 0.01.
        beta : float, optional
            Exponential decay rate for the moving average of squared gradients, by default 0.9.
        """
        super().__init__(learning_rate)
        self.beta = beta
    
    
    def update(self, layer: Layer) -> None:
        """
        Updates the weights and biases of the provided layer using the RMSProp algorithm.

        For each parameter, this method computes the moving average of squared gradients and 
        adjusts the learning rate for that particular parameter based on this average. 
        The method normalizes the gradient and uses the normalized learning rate for weight updates.

        Parameters
        ----------
        layer : Layer
            The layer whose weights and biases need to be updated.

        Returns
        -------
        None
        """
        epsilon = 1e-15
        self._update_v(layer)
        
        gradient_normalization_weights = np.sqrt(layer.vweights) + epsilon # normalizing the gradient for weights
        normalized_learning_rate = self.learning_rate / gradient_normalization_weights # normalizing the learning rate
        layer.weights -= normalized_learning_rate * layer.dweights # updating the weights
        
        if layer.has_bias:
            gradient_normalization_biases = np.sqrt(layer.vbiases) + epsilon # normalizing the gradient for biases
            normalized_learning_rate = self.learning_rate / gradient_normalization_biases # normalizing the learning rate
            layer.biases -= normalized_learning_rate * layer.dbiases # updating the biases
    
    
    def _update_v(self, layer: Layer) -> None:
        """
        Internal helper method to update the moving average of squared gradients for a layer.

        For each parameter, this method computes the moving average of squared gradients, 
        which is used in the update step to adjust the learning rate for that particular parameter.

        Parameters
        ----------
        layer : Layer
            The layer whose moving average of squared gradients need to be updated.

        Returns
        -------
        None
        """
        if not hasattr(layer, 'vweights'):
            layer.initialize_v() # initializing the vweights and vbiases attributes if they don't exist
            
        layer.vweights = self.beta * layer.vweights + (1 - self.beta) * layer.dweights ** 2 # updating the moving average of squared gradients for weights
        
        if layer.has_bias:
            layer.vbiases = self.beta * layer.vbiases + (1 - self.beta) * layer.dbiases ** 2 # updating the moving average of squared gradients for biases
