import numpy as np
from typing import List, Tuple
from skynet_ml.nn.losses import get_loss_function, get_loss_derivative

class Sequential:
    """
    Represents a sequential neural network. 
    
    Attributes:
        epochs (int): Number of epochs the model will be trained with.
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
        self.layers: list = []
        
        
    def add(self, layer) -> None:
        """
        Add a new layer to the neural network.

        Args:
            layer: The layer to be added.
        """        
        self.layers.append(layer)
        
        
    def compile(self, epochs: int, learning_rate: float, optimizer: str, batch_size: int, loss: str) -> None:
        """Configure the learning process before training starts.

        Args:
            epochs (int): Number of epochs the model will be tranied with.
            learning_rate (float): Learning rate for the optimizer.
            optimizer (str): Optimization algorithm to use.
            batch_size (int): Size of each mini batch.
            loss (str): Loss function to be used.
        """        
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.loss_function = get_loss_function(loss)
        self.loss_derivative = get_loss_derivative(loss)
        self.loss_epochs = []
        
        
    def forward(self, X: np.array) -> np.array:
        """
        Performs the forward pass through the network.

        Args:
            X (np.array): Input data.

        Returns:
        - X (np.array): yhat vector. 
        """        
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    
    def backward(self, d_output: np.array) -> None:
        """Perform the backward pass (backpropagation) through all the layers of the network

        Args:
            d_output (np.array): First term of the delta fo each layer.
        """        
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output) # gradient w.r.t input
    
            
    def optimize(self) -> None:
        """
        Updates the weights and biases for each layer.
        """        
        for layer in self.layers:
            layer.update_parameters(self.optimizer, self.learning_rate)
    
            
    def compute_loss(self, yhat: np.array, y: np.array) -> np.array:
        """
        Computes the loss of the networks.

        Args:
            yhat (np.array): Predictions (output of the network).
            y (np.array): True labels.

        Returns:
            np.array: The vector with the loss values.
        """        
        return self.loss_function(yhat, y)
    
    
    def compute_loss_derivative(self, yhat: np.array, y: np.array) -> np.array:
        """
        Computes the derivative of the loss w.r.t the predictions.

        Args:
            yhat (np.array): Predictions
            y (np.array): True Labels.

        Returns:
            np.array: The vector with the derivative of the loss values.
        """        
        return self.loss_derivative(yhat, y)
    
    
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
    
    
    def fit(self, X: np.array, y: np.array) -> None:
        """
        Train the neural network using the provided data.

        Args:
            X (np.array): Input data for training.
            y (np.array): True labels for the input data.
        """        
        for i in range(self.epochs):
            # create the mini batches for training
            mini_batches = self.create_mini_batches(X, y, self.batch_size)
            # list to store the loss for each mini batch
            loss_batches = []
            
            for batch in mini_batches:
                # unpack mini batch data and labels
                X_mini, y_mini = batch
                # forward pass: compute predictions
                yhat_mini = self.forward(X_mini)
                # compute loss of the mini batch
                loss_mini = self.compute_loss(yhat_mini, y_mini)
                # compute derivative of loss of the mini batch
                loss_derivative_mini = self.compute_loss_derivative(yhat_mini, y_mini)
                # store the loss
                loss_batches.append(loss_mini)
                # backward pass: compute the gradient of the loss w.r.t each weight and bias
                self.backward(loss_derivative_mini)
                # optimize the network's weights and bias
                self.optimize()
            
            # average loss for the epoch
            epoch_loss = np.mean(loss_batches)
            print(f"Epoch {i} Loss: {epoch_loss}")
            self.loss_epochs.append(epoch_loss)
            
            
    def predict(self, X: np.array) -> np.array:
        """
        Use the network to predict a given a given input matrix X.

        Args:
            X (np.array): Input data for prediction

        Returns:
            np.array: The predicted values.
        """        
        return self.forward(X)