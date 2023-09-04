import numpy as np


class Adam:
    """
    Adam (Adaptive Moment Estimation) is an adaptive learning rate optimization algorithm 
    designed specifically for training deep neural networks. It combines ideas from 
    RMSProp (Root Mean Square Propagation) and Momentum optimization algorithms.
    """
    
    
    def __init__(self, learning_rate: float = 0.01, beta_1: float = 0.9, beta_2: float = 0.999) -> None:
        """
        Initializes the Adam optimizer with the given hyperparameters.

        Args:
            learning_rate (float, optional): The step size used to minimize the function. Defaults to 0.01.
            beta_1 (float, optional): The exponential decay rate for the first moment estimate. Defaults to 0.9.
            beta_2 (float, optional): The exponential decay rate for the second moment estimate. Defaults to 0.999.
        """        
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = 1e-7  # prevents division by zero
        self.t = 0
    
    
    def update_params(self, layer: object) -> None:
        """
        Updates the parameters of the given layer based on the Adam optimization algorithm.
        
        For each parameter, this function computes the first and second moments 
        (moving averages) of the gradients and then adjusts the parameters using 
        an adaptive learning rate.

        Args:
            layer (object): An instance of a layer with weights and biases that need updating.
        """
        # assert 
        assert hasattr(layer, 'weights') and hasattr(layer, 'd_weights'), "Layer must have 'weights' and 'd_weights' attributes."
        
        # Initialize moving averages of the gradients and squared gradients if they don't exist
        if not hasattr(layer, "m_weights"):
            layer.initialize_velocity()
            layer.initialize_momentum()
        
        # Increment timestep
        self.t += 1
        
        # Compute the moving averages of the gradients and squared gradients
        layer.m_weights = self.beta_1 * layer.m_weights + (1 - self.beta_1) * layer.d_weights
        layer.v_weights = self.beta_2 * layer.v_weights + (1 - self.beta_2) * layer.d_weights**2
        
        # Correct the moving averages for initialization bias
        m_weights_corrected = layer.m_weights / (1 - self.beta_1**self.t)
        v_weights_corrected = layer.v_weights / (1 - self.beta_2**self.t)
        
        # Adjust the parameters using the adaptive learning rate
        layer.weights -= self.learning_rate * m_weights_corrected / (np.sqrt(v_weights_corrected) + self.epsilon)
        
        if layer.has_bias:
            # Compute the moving averages of the biases' gradients and squared gradients
            layer.m_bias = self.beta_1 * layer.m_bias + (1 - self.beta_1) * layer.d_bias
            layer.v_bias = self.beta_2 * layer.v_bias + (1 - self.beta_2) * layer.d_bias**2
            
            # Correct the moving averages for initialization bias
            m_bias_corrected = layer.m_bias / (1 - self.beta_1**self.t)
            v_bias_corrected = layer.v_bias / (1 - self.beta_2**self.t)
            
            # Adjust the bias using the adaptive learning rate
            layer.bias -= self.learning_rate * m_bias_corrected / (np.sqrt(v_bias_corrected) + self.epsilon)
            
            
    def step(self, layers: list) -> None:
        """
        Performs an optimization step for a sequence of layers using the Adam optimization algorithm.
        
        It iterates through each layer in the sequence and updates its parameters using the Adam update rule.

        Args:
            layers (list): A list of layer instances whose parameters need updating.
        """
        for layer in layers:
            self.update_params(layer)
