import numpy as np

def initialize_weights(method: str, input_size: int, output_size: int) -> np.array:
    """
    Initializes the weights based on the initialization method.

    Args:
        method (str): Initialization method.
        input_size (int): Number of columns in the weight matrix.
        output_size (int): Number of rows in the weight matrix.

    Returns:
        np.array: weight matrix.
    """    
    if method == 'random':
        return np.random.randn(output_size, input_size)


def initialize_bias(has_bias: bool, method: str, output_size: int) -> np.array:
    """
    Initializes the bias based based on the initialization method. 

    Args:
        has_bias (bool): If bias should have value or not.
        method (str): Initialization method.
        output_size (int): Number of values in bias vector.

    Returns:
        np.array: Bias vector.
    """    
    if has_bias:
        if method == "random":
            return np.random.rand(output_size, 1)
            
    return np.zeros((output_size, 1))
