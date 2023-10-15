from skynet_ml.nn.optimizers.base import BaseOptimizer
from skynet_ml.nn.layers.base import BaseLayer

class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.

    SGD is a simple yet effective optimization algorithm used to minimize the loss 
    function in iterative manners. It updates the network's parameters (weights and biases) 
    in the opposite direction of the gradient.
    """    
    
    
    def __init__(self, learning_rate: float = 0.01) -> None:
        """
        Initializes the SGD optimizer with a specified learning rate.

        Args:
            learning_rate (float, optional): The step size used to update the parameters 
            during training. Defines how large the updates to the weights/biases will be 
            on each iteration. Defaults to 0.01.
        """        
        super().__init__(learning_rate)
        self.name = "sgd"


    def update(self, layer: BaseLayer) -> None:
        """
        Performs a parameter update using the computed gradients.

        Adjusts the weights and biases (if present) of the given layer using the 
        simple SGD formula.

        Args:
            layer (BaseLayer): The neural network layer whose parameters (weights and biases) 
            need to be updated based on the computed gradients.
        """
        
        # updating the weights
        layer.weights -= self.learning_rate * layer.d_weights
        
        # updating the bias
        if layer.has_bias:
            layer.bias -= self.learning_rate * layer.d_bias