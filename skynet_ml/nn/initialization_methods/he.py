from skynet_ml.nn.initialization_methods.initializer import Initializer
import numpy as np

class HeNormalInitializer(Initializer):
    """
    Initializer that generates weights from a normal distribution based on He initialization.

    This initialization method is specially designed for layers with ReLU activations, 
    and it aims to keep the variance of the outputs of a layer to be the same as 
    the variance of its inputs.

    Methods
    -------
    initialize_weights(input_dim: int, n_units: int) -> np.array:
        Generate and return weights initialized from a normal distribution 
        with mean of 0 and variance 2 divided by the number of input units.

    """
    
    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Generate weights from a normal distribution based on He initialization.

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
        std = np.sqrt(2.0 / input_dim)
        return np.random.normal(0.0, std, (input_dim, n_units))


class HeUniformInitializer(Initializer):
    """
    Initializer that generates weights uniformly based on He initialization.

    Similar to HeNormalInitializer, this method is designed for layers with ReLU activations 
    to help keep the variance of the outputs of a layer to be the same as 
    the variance of its inputs. However, weights are initialized from a uniform distribution.

    Methods
    -------
    initialize_weights(input_dim: int, n_units: int) -> np.array:
        Generate and return weights initialized uniformly based on He initialization.

    """
    
    def initialize_weights(self, input_dim: int, n_units: int) -> np.array:
        """
        Generate weights uniformly based on He initialization.

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
        limit = np.sqrt(6.0 / input_dim)
        return np.random.uniform(-limit, limit, (input_dim, n_units))
