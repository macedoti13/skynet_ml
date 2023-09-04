import numpy as np

class AdaGrad:
    """
    AdaGrad (Adaptive (per parameter) gradient) optimizer. 
    
    The idea of AdaGrad is to give bigger steps for parameters that are rarely used and small steps
    for parameters that are frequently updated.
    """
    
    def __init__(self, learning_rate: float=0.01) -> None:
        """
        Initializes the AdaGrad optimizer with a specified learning rate.
        
        AdaGrad adapts the learning rate for each parameter during training, 
        which can help in scenarios where different parameters have different
        sensitivities in the loss function.

        Args:
            learning_rate (float, optional): The initial learning rate. Defaults to 0.01.
        """        
        self.learning_rate = learning_rate
        self.epsilon = 1e-7 # prevents division by zero
        
        
    def update_params(self, layer: object) -> None:
        """
        Updates the parameters of the given layer using the AdaGrad optimization algorithm.
        
        For each parameter, it adjusts the learning rate based on the historical squared gradient 
        for that parameter, ensuring that frequently updated parameters get smaller learning rates.

        Args:
            layer (object): An instance of a layer, which has weights and biases that need updating.
        """        
        # assert
        assert hasattr(layer, 'weights') and hasattr(layer, 'd_weights'), "Layer must have 'weights' and 'd_weights' attributes."
        
        # If 'v_weights' and 'v_bias' are not initialized, initialize them
        if not hasattr(layer, "v_weights"):
            layer.initialize_velocity()
            
        # accumulate squared gradients
        layer.v_weights += layer.d_weights**2
        if layer.has_bias:
            layer.v_bias += layer.d_bias**2
            
        # update weights and bias
        adjusted_gradient_weights = np.sqrt(layer.v_weights) + self.epsilon
        layer.weights -= self.learning_rate / adjusted_gradient_weights * layer.d_weights
        
        if layer.has_bias:
            adjusted_gradient_bias = np.sqrt(layer.v_bias) + self.epsilon
            layer.bias -= self.learning_rate / adjusted_gradient_bias * layer.d_bias
        
    
    def step(self, layers: list) -> None:
        """
        Executes an optimization step for a list of layers using AdaGrad.
        
        It iterates through each layer in the list and updates its parameters 
        using the AdaGrad update rule.

        Args:
            layers (list): A list of layer instances whose parameters need updating.
        """        
        for layer in layers:
            self.update_params(layer)
