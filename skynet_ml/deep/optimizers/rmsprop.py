import numpy as np

class RMSProp:
    """
    RMSProp (Root Mean Square Propagation) Optimizer.
    
    It improves AdaGrad by applying a decay value, so the effective learning rate doesn't get extremely low. RMSProp basically keeps a exponencial moving 
    avarge of the squared of the past gradients. 
    """    
    
    
    def __init__(self, learning_rate: float = 0.01, beta: float = 0.9) -> None:
        """
        Initializes the RMSProp optimizer with a specified learning rate and decay value.

        Args:
            learning_rate (float, optional): The initial learning rage. Defaults to 0.01.
            beta (float, optional): Value of decaying for our normalized gradients. Defaults to 0.9.
        """        
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = 1e-7 # prevents division by zero
        
        
    def update_params(self, layer: object) -> None: 
        """
        Updates the parameters of the given layer using the RMSProp optimizer. 
        
        For each parameter, it adjusts the learning rate based on a exponencialy decreasing moving average
        of the past squared gradients. Ensuring that the learning rate won't get very small after a given number 
        of iteractions.

        Args:
            layer (object): An instance of a layer, which has weights and biases that need updating.
        """        
        # assert 
        assert hasattr(layer, 'weights') and hasattr(layer, 'd_weights'),  "Layer must have 'weights' and 'd_weights' attributes."
        
        # if 'v_weights' and 'v_bias' are not initialized, initialize them
        if not hasattr(layer, "v_weights"):
            layer.initialize_velocity()
            
        # update accumulated squared gradients
        layer.v_weights = self.beta * layer.v_weights + (1 - self.beta) * layer.d_weights**2
        if layer.has_bias:
            layer.v_bias = self.beta * layer.v_bias + (1 - self.beta) * layer.d_bias**2
            
        # update weights and bias
        adjusted_gradient_weights = np.sqrt(layer.v_weights) + self.epsilon
        layer.weights -= self.learning_rate / adjusted_gradient_weights * layer.d_weights
        
        if layer.has_bias:
            adjusted_gradient_bias = np.sqrt(layer.v_bias) + self.epsilon
            layer.bias -= self.learning_rate / adjusted_gradient_bias * layer.d_bias
            

    def step(self, layers: list) -> None:
        """
        Executes an optimization step for a list of layers using RMSProp.
        
        It iterates through each layer in the list and updates its parameters 
        using the RMSProp update rule.

        Args:
            layers (list): A list of layer instances whose parameters need updating.
        """       
        for layer in layers:
            self.update_params(layer)
        