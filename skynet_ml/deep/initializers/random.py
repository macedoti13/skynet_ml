import numpy as np

class Random:
    """
    Initializer that uses random values for weights and biases.
    
    This initializer uses a standard normal distribution for weights and 
    a uniform distribution over [0, 1) for biases.
    """    
    
    def initialize_weights(self, input_dim: int, output_dim: int) -> np.array:
        """
        Initializes the weights with random values drawn from a standard normal distribution.

        Args:
            input_dim (int): Dimension of the input data or number of input neurons.
            output_dim (int): Number of output neurons.

        Returns:
            np.array: A 2D array of shape (output_dim, input_dim) initialized with random values.
        """        
        return np.random.randn(output_dim, input_dim)
    
    
    def initialize_bias(self, output_dim: int, has_bias: bool = True) -> np.array:
        """
        Initializes the bias with random values drawn from a uniform distribution over [0, 1).
        If has_bias is False, the bias is initialized with zeros.

        Args:
            output_dim (int): Number of output neurons.
            has_bias (bool): Determines whether to include a bias term. Defaults to True.

        Returns:
            np.array: A 2D array of shape (output_dim, 1) initialized either with random values or zeros.
        """        
        if has_bias:
            return np.random.rand(output_dim, 1)
        
        return np.zeros((output_dim, 1))
