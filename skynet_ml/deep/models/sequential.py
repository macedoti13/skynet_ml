import numpy as np
from skynet_ml.deep.losses import loss_map, d_loss_map
from skynet_ml.deep.optimizers import optimizer_map
from typing import List, Tuple


class Sequential:
    """
    Represents a feed forward neural network. 
    
    Attributes:
        learning_rate (float): Learning rate for the optimizer.
        optimizer (str): Optimization algorithm to use.
        batch_size (int): Size of each mini_batch.
        loss_function (function): Loss function to be used.
        loss_derivative (function): Derivative of the loss function.
        loss_epochs (list): List with the loss for each epoch.
    """
        
        
    def __init__(self) -> None:
        """
        Initializes the Sequential model.
        
        Attributes: 
        - layers: List of layers added to the neural network.
        """        
        self.layers = []
        
        
    def add(self, layer: object) -> None:
        """
        Add a new layer to the neural network.

        Args:
            layer: The layer to be added.
        """       
         
        assert hasattr(layer, 'forward') and hasattr(layer, 'backward'), "Layer must have 'forward' and 'backward' methods."

        self.layers.append(layer)
        
        
    def compile(self, optimizer="sgd", learning_rate: float = 0.01, loss: str="mse"):
        """
        Configure the network after all layers are added.

        Args:
            optimizer (str, optional): Optimization algorithm. Defaults to "sgd".
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.01.
            loss (str, optional): Loss function to be used. Defaults to "mse".
        """   
             
        assert optimizer in optimizer_map or hasattr(optimizer, 'step'), "Invalid optimizer type."
        assert loss in loss_map, "Invalid loss type."

        # get's the optimizer from the str or just assigns the object given
        if isinstance(optimizer, str):
            self.optimizer = optimizer_map[optimizer]
            self.optimizer.learning_rate = learning_rate
        else:
            self.optimizer = optimizer
            
        self.loss = loss_map[loss]
        self.d_loss = d_loss_map[loss]
        self.loss_epochs = []
    
    
    def forward(self, X: np.array) -> np.array:
        """
        Performs the forward pass through the network.

        Args:
            X (np.array): Input data.

        Returns:
        - X (np.array): yhat vector. 
        """   
           
        assert len(self.layers) > 0, "No layers added to the model."

        for layer in self.layers:
            X = layer.forward(X)
            
        return X
    
    
    def backward(self, d_output: np.array) -> np.array:
        """
        Performs the backward pass (backpropagation) through all the layers of the network

        Args:
            d_output (np.array): First term of the delta fo each layer.
        """ 
        
        assert hasattr(self.layers[-1], 'z'), "Forward method must be called before backward."

        for layer in reversed(self.layers):
            d_output = layer.backward(d_output)
            
        return d_output
    
    
    def compute_loss(self, yhat: np.array, y: np.array) -> np.array:
        """
        Computes the loss of the networks.

        Args:
            yhat (np.array): Predictions (output of the network).
            y (np.array): True labels.

        Returns:
            np.array: The vector with the loss values.
        """        
        return self.loss(yhat, y)
    
    
    def compute_d_loss(self, yhat: np.array, y: np.array) -> np.array:
        """
        Computes the derivative of the loss w.r.t the predictions.

        Args:
            yhat (np.array): Predictions
            y (np.array): True Labels.

        Returns:
            np.array: The vector with the derivative of the loss values.
        """        
        return self.d_loss(yhat, y)
    
    
    def create_mini_batches(self, X: np.array, y: np.array, batch_size: int) -> List[Tuple[np.array, np.array]]:
        """
        Create mini-batches from the provided data.

        Args:
        - X: Input data.
        - y: True labels.
        - batch_size: Desired size of each mini-batch.

        Returns:
        - mini_batches: List of tuples, where each tuple contains a mini-batch of data and corresponding labels.
        """
        
        assert X.shape[1] == y.shape[1], "Mismatch between the number of samples in X and y."

        # Create an array of indices from 0 to the number of samples.
        indices = np.arange(X.shape[1])
        np.random.shuffle(indices)  # Shuffle the indices.
        X = X[:, indices]  # Shuffle X using the shuffled indices.
        y = y[:, indices]  # Shuffle y using the shuffled indices.

        mini_batches = []

        total_batches = X.shape[1] // batch_size
        for i in range(total_batches):
            X_mini = X[:, i * batch_size: (i + 1) * batch_size]
            y_mini = y[:, i * batch_size: (i + 1) * batch_size]
            mini_batches.append((X_mini, y_mini))

        # Handle the end case (last mini-batch < mini_batch_size)
        if X.shape[1] % batch_size != 0:
            X_mini = X[:, total_batches * batch_size:]
            y_mini = y[:, total_batches * batch_size:]
            mini_batches.append((X_mini, y_mini))

        return mini_batches
    
    
    def fit(self, x_train: np.array, y_train: np.array, epochs: int = 1, batch_size: int = 16): 
        """
        Train the neural network using the provided data.

        Args:
            X (np.array): Input data for training.
            y (np.array): True labels for the input data.
        """      
        # asserts
        assert x_train.shape[1] == y_train.shape[1], "Mismatch between the number of samples in x_train and y_train."
        assert epochs > 0, "Number of epochs should be greater than 0."
        assert batch_size > 0, "Batch size should be greater than 0."

        # for each epoch 
        for i in range(epochs):
            
            # create mini-batches for training
            mini_batches = self.create_mini_batches(x_train, y_train, batch_size)
            # list to store the loss for each mini-batch
            loss_batches = []
            
            # for every mini-batch
            for batch in mini_batches:
                
                # unpack mini-batch data and labels
                x_mini, y_mini = batch
                
                # forward pass with mini-batch
                yhat_mini = self.forward(x_mini)
                
                # compute the loss for the mini_batch and save it
                loss_mini = self.compute_loss(yhat_mini, y_mini)
                loss_batches.append(loss_mini)
                
                # compute the derivative of the loss for the mini-batch
                d_loss_mini = self.compute_d_loss(yhat_mini, y_mini)
                
                # backward pass: compute the gradient of the loss w.r.t each weight and bias
                self.backward(d_loss_mini)
                
                # optimizes the weigths in every layer
                self.optimizer.step(self.layers)
                
            # calculate and print average loss of epoch
            loss_epoch = np.mean(loss_batches)
            self.loss_epochs.append(loss_epoch)
            print(f"Epoch {i}/{epochs} - Loss: {loss_epoch}")
            
            
    def predict(self, X: np.array) -> np.array:
            """
            Use the network to predict a given a given input matrix X.

            Args:
                X (np.array): Input data for prediction

            Returns:
                np.array: The predicted values.
            """        
            # aseert
            assert len(self.layers) > 0, "No layers added to the model."

            return self.forward(X)