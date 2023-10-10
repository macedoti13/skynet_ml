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
        Updates the weights and bias of the provided layer using the Adam algorithm.

    _update_m(layer: Layer) -> None:
        Internal helper method to update the moving average of gradients for a layer.

    _update_v(layer: Layer) -> None:
        Internal helper method to update the moving average of squared gradients for a layer.

    _correct_m_bias(layer: Layer) -> float:
        Compute bias-corrected moving average of gradients for bias.

    _correct_m_weights(layer: Layer) -> float:
        Compute bias-corrected moving average of gradients for weights.

    _correct_v_bias(layer: Layer) -> float:
        Compute bias-corrected moving average of squared gradients for bias.

    _correct_v_weights(layer: Layer) -> float:
        Compute bias-corrected moving average of squared gradients for weights.
    """
    
    
    def __init__(self, learning_rate: float = 0.01, beta1: float = 0.9, beta2: float = 0.999) -> None:
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0 # timestep
        
        
    def get_config(self) -> dict:
        """
        Returns a dictionary containing the configuration of the optimizer.

        Returns
        -------
        dict
            A dictionary containing the configuration of the optimizer.
        """
        return {
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2
        }
        
        
    def update(self, layer: Layer) -> None: 
        """
        Update the weights and bias of the provided layer using the Adam optimization algorithm.

        Parameters
        ----------
        layer : Layer
            The neural network layer whose weights and bias need to be updated.
        """
        epsilon = 1e-15 
        self.t += 1 # increment timestep
        
        self._update_m(layer) # update m for weights and bias
        self._update_v(layer) # update v for weights and bias
        
        m_weights_corrected = self._correct_m_weights(layer) # correct m for weights 
        v_weights_corrected = self._correct_v_weights(layer) # correct v for weights
        
        layer.weights -= self.learning_rate * m_weights_corrected / (np.sqrt(v_weights_corrected) + epsilon) # update weights
        
        if layer.has_bias:
            m_bias_corrected = self._correct_m_bias(layer) # correct m for bias
            v_bias_corrected = self._correct_v_bias(layer) # correct v for bias
            
            layer.bias -= self.learning_rate * m_bias_corrected / (np.sqrt(v_bias_corrected) + epsilon) # update bias
            
            
    def _update_m(self, layer: Layer) -> None:
        """
        Update the moving average of gradients (m) for the provided layer.

        Parameters
        ----------
        layer : Layer
            The neural network layer for which the moving average of gradients is updated.
        """
        
        if not hasattr(layer, 'm_weights'):
            layer.initialize_momentum() # initialize m_weights and m_bias if they don't exist
            
        layer.m_weights = self.beta1 * layer.m_weights + (1 - self.beta1) * layer.d_weights # update m for weights
        
        if layer.has_bias:
            layer.m_bias = self.beta1 * layer.m_bias + (1 - self.beta1) * layer.m_bias  # update m for bias
            
            
    def _update_v(self, layer: Layer) -> None:
        """
        Update the moving average of squared gradients (v) for the provided layer.

        Parameters
        ----------
        layer : Layer
            The neural network layer for which the moving average of squared gradients is updated.
        """
        
        if not hasattr(layer, 'v_weights'):
            layer.initialize_velocity()
            
        layer.v_weights = self.beta2 * layer.v_weights + (1 - self.beta2) * layer.d_weights ** 2 # update v for weights
        
        if layer.has_bias:
            layer.v_bias = self.beta2 * layer.v_bias + (1 - self.beta2) * layer.d_bias ** 2 # update v for bias
            
            
    def _correct_m_bias(self, layer: Layer) -> None:
        """
        Compute and return the bias-corrected moving average of gradients for bias of the provided layer.

        Parameters
        ----------
        layer : Layer
            The neural network layer for which the bias-corrected moving average of gradients for bias is computed.

        Returns
        -------
        float
            The bias-corrected moving average of gradients for bias.
        """
        return layer.m_bias / (1 - self.beta1 ** self.t) # correct m for bias
    
    
    def _correct_m_weights(self, layer: Layer) -> None:
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
        return layer.m_weights / (1 - self.beta1 ** self.t) # correct m for weights
    
    
    def _correct_v_bias(self, layer: Layer) -> None:
        """
        Compute and return the bias-corrected moving average of squared gradients for bias of the provided layer.

        Parameters
        ----------
        layer : Layer
            The neural network layer for which the bias-corrected moving average of squared gradients for bias is computed.

        Returns
        -------
        float
            The bias-corrected moving average of squared gradients for bias.
        """
        return layer.v_bias / (1 - self.beta2 ** self.t) # correct v for bias
    
    
    def _correct_v_weights(self, layer: Layer) -> None:
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
        return layer.v_weights / (1 - self.beta2 ** self.t) # correct v for weights
