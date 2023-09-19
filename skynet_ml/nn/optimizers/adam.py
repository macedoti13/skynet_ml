from skynet_ml.nn.optimizers.optimizer import Optimizer
from skynet_ml.nn.layers.layer import Layer
import numpy as np

class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) optimization algorithm.

    Adam is an adaptive learning rate optimization algorithm that computes adaptive learning rates
    for each parameter. It uses both the moving average of past gradients (m) and the moving average
    of past squared gradients (v) to adjust the weights during training.

    Attributes
    ----------
    beta1 : float
        Exponential decay rate for the moving average of gradients.
    beta2 : float
        Exponential decay rate for the moving average of squared gradients.
    t : int
        Time step which gets incremented after each batch.

    Methods
    -------
    update(layer: Layer) -> None:
        Updates the weights and biases of the provided layer using the Adam algorithm.

    _update_m(layer: Layer) -> None:
        Internal helper method to update the moving average of gradients for a layer.

    _update_v(layer: Layer) -> None:
        Internal helper method to update the moving average of squared gradients for a layer.

    _correct_mbiases(layer: Layer) -> float:
        Compute bias-corrected moving average of gradients for biases.

    _correct_mweights(layer: Layer) -> float:
        Compute bias-corrected moving average of gradients for weights.

    _correct_vbiases(layer: Layer) -> float:
        Compute bias-corrected moving average of squared gradients for biases.

    _correct_vweights(layer: Layer) -> float:
        Compute bias-corrected moving average of squared gradients for weights.
    """
    
    def __init__(self, learning_rate: float = 0.01, beta1: float = 0.9, beta2: float = 0.999) -> None:
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0 # timestep
        
        
    def update(self, layer: Layer) -> None: 
        """
        Update the weights and biases of the provided layer using the Adam optimization algorithm.

        Parameters
        ----------
        layer : Layer
            The neural network layer whose weights and biases need to be updated.
        """
        epsilon = 1e-15 
        self.t += 1 # increment timestep
        
        self._update_m(layer) # update m for weights and biases
        self._update_v(layer) # update v for weights and biases
        
        mweights_corrected = self._correct_mweights(layer) # correct m for weights 
        vweights_corrected = self._correct_vweights(layer) # correct v for weights
        
        layer.weights -= self.learning_rate * mweights_corrected / (np.sqrt(vweights_corrected) + epsilon) # update weights
        
        if layer.has_bias:
            mbiases_corrected = self._correct_mbiases(layer) # correct m for biases
            vbiases_corrected = self._correct_vbiases(layer) # correct v for biases
            
            layer.biases -= self.learning_rate * mbiases_corrected / (np.sqrt(vbiases_corrected) + epsilon) # update biases
            
            
    def _update_m(self, layer: Layer) -> None:
        """
        Update the moving average of gradients (m) for the provided layer.

        Parameters
        ----------
        layer : Layer
            The neural network layer for which the moving average of gradients is updated.
        """
        
        if not hasattr(layer, 'mweights'):
            layer.initialize_m() # initialize mweights and mbiases if they don't exist
            
        layer.mweights = self.beta1 * layer.mweights + (1 - self.beta1) * layer.dweights # update m for weights
        
        if layer.has_bias:
            layer.mbiases = self.beta1 * layer.mbiases + (1 - self.beta1) * layer.mbiases  # update m for biases
            
            
    def _update_v(self, layer: Layer) -> None:
        """
        Update the moving average of squared gradients (v) for the provided layer.

        Parameters
        ----------
        layer : Layer
            The neural network layer for which the moving average of squared gradients is updated.
        """
        
        if not hasattr(layer, 'vweights'):
            layer.initialize_v()
            
        layer.vweights = self.beta2 * layer.vweights + (1 - self.beta2) * layer.dweights ** 2 # update v for weights
        
        if layer.has_bias:
            layer.vbiases = self.beta2 * layer.vbiases + (1 - self.beta2) * layer.dbiases ** 2 # update v for biases
            
            
    def _correct_mbiases(self, layer: Layer) -> None:
        """
        Compute and return the bias-corrected moving average of gradients for biases of the provided layer.

        Parameters
        ----------
        layer : Layer
            The neural network layer for which the bias-corrected moving average of gradients for biases is computed.

        Returns
        -------
        float
            The bias-corrected moving average of gradients for biases.
        """
        return layer.mbiases / (1 - self.beta1 ** self.t) # correct m for biases
    
    
    def _correct_mweights(self, layer: Layer) -> None:
        """
        Compute and return the bias-corrected moving average of gradients for weights of the provided layer.

        Parameters
        ----------
        layer : Layer
            The neural network layer for which the bias-corrected moving average of gradients for weights is computed.

        Returns
        -------
        float
            The bias-corrected moving average of gradients for weights.
        """
        return layer.mweights / (1 - self.beta1 ** self.t) # correct m for weights
    
    
    def _correct_vbiases(self, layer: Layer) -> None:
        """
        Compute and return the bias-corrected moving average of squared gradients for biases of the provided layer.

        Parameters
        ----------
        layer : Layer
            The neural network layer for which the bias-corrected moving average of squared gradients for biases is computed.

        Returns
        -------
        float
            The bias-corrected moving average of squared gradients for biases.
        """
        return layer.vbiases / (1 - self.beta2 ** self.t) # correct v for biases
    
    
    def _correct_vweights(self, layer: Layer) -> None:
        """
        Compute and return the bias-corrected moving average of squared gradients for weights of the provided layer.

        Parameters
        ----------
        layer : Layer
            The neural network layer for which the bias-corrected moving average of squared gradients for weights is computed.

        Returns
        -------
        float
            The bias-corrected moving average of squared gradients for weights.
        """
        return layer.vweights / (1 - self.beta2 ** self.t) # correct v for weights
