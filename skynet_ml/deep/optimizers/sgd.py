class SGD:
    """
    Stochastic Gradient Descent Algorithm. 
    """    

    def __init__(self, learning_rate: float=0.01) -> None:
        """
        Init method for the stochastic gradient descent algorithm.

        Args:
            learning_rate (float, optional): Size of the step that will be taken by the optimizer. Defaults to 0.01.
        """        
        self.learning_rate = learning_rate
        
        
    def update_params(self, layer: object) -> None:
        """
        Updates the parameters for a given layer by applying the gradient descent update rule. 

        Args:
            layer (object): Layer of the network.
        """        
        # assert
        assert hasattr(layer, 'weights') and hasattr(layer, 'd_weights'), "Layer must have 'weights' and 'd_weights' attributes."

        # updates the weights for a given layer -> w_(t+1) = w_(t) - alpha * gradient
        layer.weights -= self.learning_rate * layer.d_weights
        
        # updates the bias for a given layer -> b_(t+1) = b_(t) - alpha * gradient
        if layer.bias is not None:
            layer.bias -= self.learning_rate * layer.d_bias
            
            
    def step(self, layers: list) -> None:
        """
        Updates the params for every single layer in layers. 

        Args:
            layers (list): List with all the layers in the network.
        """        
        for layer in layers:
            self.update_params(layer)

