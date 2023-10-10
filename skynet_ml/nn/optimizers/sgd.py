from skynet_ml.nn.optimizers.optimizer import Optimizer
from skynet_ml.nn.layers.layer import Layer

class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimization algorithm.

    SGD is an iterative method to optimize the objective function with suitable smoothness 
    properties. This optimizer updates each parameter of the model in the opposite direction 
    of the gradient of the objective function with respect to that parameter, scaled by 
    the learning rate. Proper use of SGD can lead to faster convergence compared to other 
    optimization techniques.

    Inherits from:
    Optimizer: Base class for optimization algorithms.

    Attributes
    ----------
    - learning_rate : float
        The step size used when following the negative gradient during training.

    Methods
    -------
    - update(layer: Layer) -> None:
        Updates the weights and biases of the provided layer using the SGD optimization 
        algorithm.

    - step(layers: list) -> None:
        Updates the weights and biases for each layer in the list using the SGD optimization 
        algorithm.
    """
    
    
    def update(self, layer: Layer) -> None:
        """
        Updates the weights of the provided layer using the SGD optimization algorithm.

        Parameters:
        - layer (Layer): The layer whose weights and biases need to be updated.

        Note:
        If the layer has biases, they will be updated as well.
        """
        layer.weights -= self.learning_rate * layer.d_weights
        
        if layer.has_bias:
            layer.bias -= self.learning_rate * layer.d_bias


    def get_config(self) -> dict:
        """
        Returns a dictionary containing the configuration of the optimizer.

        Returns:
        - config (dict): Configuration of the optimizer.
        """
        return {'learning_rate': self.learning_rate}