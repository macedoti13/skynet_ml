from skynet_ml.nn.optimizers.base import BaseOptimizer
from skynet_ml.nn.layers.base import BaseLayer
import numpy as np


class Adam(BaseOptimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.

    Adam is an optimization algorithm that can handle sparse gradients on noisy problems. 
    It combines the advantages of both AdaGrad and RMSProp, by using moving averages of both 
    the gradients and the squared gradients.
    """    


    def __init__(self, learning_rate: float = 0.01, beta1: float = 0.9, beta2: float = 0.999) -> None:
        """
        Initializes the Adam optimizer with the given hyperparameters.

        Args:
            learning_rate (float, optional): The step size used to update the parameters. Defaults to 0.01.
            beta1 (float, optional): Exponential decay rate for the first moment estimates. Defaults to 0.9.
            beta2 (float, optional): Exponential decay rate for the second moment estimates. Defaults to 0.999.
        """  
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0 # timestep
        self.name = f"adam_{str(self.learning_rate)}_{str(self.beta1)}_{str(self.beta2)}"
        
        
        
    def update(self, layer: BaseLayer) -> None: 
        """
        Performs a parameter update using the Adam algorithm.

        Args:
            layer (BaseLayer): The neural network layer whose parameters (weights and biases) 
                               need to be updated.
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
            
            
            
    def _update_m(self, layer: BaseLayer) -> None:
        """
        Updates the moving average of gradients for the given layer.

        Args:
            layer (BaseLayer): The neural network layer for which the moving average of gradients is being updated.
        """    
        
        if not hasattr(layer, 'm_weights'):
            layer.initialize_momentum() # initialize m_weights and m_bias if they don't exist
            
        layer.m_weights = self.beta1 * layer.m_weights + (1 - self.beta1) * layer.d_weights # update m for weights
        
        if layer.has_bias:
            layer.m_bias = self.beta1 * layer.m_bias + (1 - self.beta1) * layer.m_bias  # update m for bias
            
            
            
    def _update_v(self, layer: BaseLayer) -> None:
        """
        Updates the moving average of squared gradients for the given layer.

        Args:
            layer (BaseLayer): The neural network layer for which the moving average of squared gradients 
                               is being updated.
        """
        
        if not hasattr(layer, 'v_weights'):
            layer.initialize_velocity()
            
        layer.v_weights = self.beta2 * layer.v_weights + (1 - self.beta2) * layer.d_weights ** 2 # update v for weights
        
        if layer.has_bias:
            layer.v_bias = self.beta2 * layer.v_bias + (1 - self.beta2) * layer.d_bias ** 2 # update v for bias
            
            
            
    def _correct_m_bias(self, layer: BaseLayer) -> float:
        """
        Returns the bias-corrected first moment estimate for the layer's biases.

        Args:
            layer (BaseLayer): The neural network layer for which the bias-corrected first moment estimate 
                               for the biases is being computed.

        Returns:
            float: Bias-corrected first moment estimate for the layer's biases.
        """
        return layer.m_bias / (1 - self.beta1 ** self.t) # correct m for bias
    
    
    
    def _correct_m_weights(self, layer: BaseLayer) -> float:
        """
        Returns the bias-corrected first moment estimate for the layer's weights.

        Args:
            layer (BaseLayer): The neural network layer for which the bias-corrected first moment estimate 
                               for the weights is being computed.

        Returns:
            float: Bias-corrected first moment estimate for the layer's weights.
        """
        return layer.m_weights / (1 - self.beta1 ** self.t) # correct m for weights
    
    
    
    def _correct_v_bias(self, layer: BaseLayer) -> float:
        """
        Returns the bias-corrected second raw moment estimate for the layer's biases.

        Args:
            layer (BaseLayer): The neural network layer for which the bias-corrected second raw moment estimate 
                               for the biases is being computed.

        Returns:
            float: Bias-corrected second raw moment estimate for the layer's biases.
        """
        return layer.v_bias / (1 - self.beta2 ** self.t) # correct v for bias
    
    
    
    def _correct_v_weights(self, layer: BaseLayer) -> float:
        """
        Returns the bias-corrected second raw moment estimate for the layer's weights.

        Args:
            layer (BaseLayer): The neural network layer for which the bias-corrected second raw moment estimate 
                               for the weights is being computed.

        Returns:
            float: Bias-corrected second raw moment estimate for the layer's weights.
        """
        return layer.v_weights / (1 - self.beta2 ** self.t) # correct v for weights
