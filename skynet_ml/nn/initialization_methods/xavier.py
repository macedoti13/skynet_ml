from skynet_ml.nn.initialization_methods.initializer import Initializer
import numpy as np

class XavierNormalInitializer(Initializer):
    """
    Initializer that generates weights from a normal distribution based on Xavier (Glorot) initialization.

    This initialization method is designed to keep the scale of the gradients roughly the same 
    in all layers. Specifically for the case where the activation function is a hyperbolic tangent 
    (tanh), but it often works well for other activation functions too.

    Methods
    -------
    initialize_weights(input_dim: int, n_units: int) -> np.array:
        Generate and return weights initialized from a normal distribution 
        with mean of 0 and variance 2 divided by the number of input units.

    """
    def __init__(self) -> None:
        """
        Initialize the XavierNormalInitializer class.
        """
        self.name = "Xavier Normal"
        
    
    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Generate weights from a normal distribution based on Xavier initialization.

        Parameters
        ----------
        input_dim : int
            The number of input features or neurons.
        
        n_units : int
            The number of units or neurons in the layer.

        Returns
        -------
        np.array
            A 2D numpy array of initialized weights with shape (input_dim, n_units).

        """
        std = np.sqrt(1.0 / input_dim)
        return np.random.normal(0.0, std, (input_dim, n_units))


class XavierUniformInitializer(Initializer):
    """
    Initializer that generates weights uniformly based on Xavier (Glorot) initialization.

    Similar to XavierNormalInitializer, this method is designed to keep the scale 
    of the gradients roughly the same in all layers. However, weights are initialized 
    from a uniform distribution ranging from -sqrt(6/(n_inputs + n_units)) to sqrt(6/(n_inputs + n_units)).

    Methods
    -------
    initialize_weights(input_dim: int, n_units: int) -> np.array:
        Generate and return weights initialized uniformly based on Xavier initialization.

    """
    def __init__(self) -> None:
        """
        Initialize the XavierUniformInitializer class.
        """
        self.name = "Xavier Uniform"
        
    
    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Generate weights uniformly based on Xavier initialization.

        Parameters
        ----------
        input_dim : int
            The number of input features or neurons.
        
        n_units : int
            The number of units or neurons in the layer.

        Returns
        -------
        np.array
            A 2D numpy array of initialized weights with shape (input_dim, n_units).

        """
        limit = np.sqrt(6.0 / (input_dim + n_units))
        return np.random.uniform(-limit, limit, (input_dim, n_units))
