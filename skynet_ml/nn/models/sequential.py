from skynet_ml.nn.layers.layer import Layer
from skynet_ml.nn.regularizers.regularizer import Regularizer
from skynet_ml.nn.optimizers.optimizer import Optimizer
from skynet_ml.nn.loss_functions.loss import Loss
from skynet_ml.metrics import METRICS_MAP
from skynet_ml.metrics.metric import Metric
from skynet_ml.utils._regularizer_factory import RegularizerFactory
from skynet_ml.utils._optimizer_factory import OptimizerFactory
from skynet_ml.utils._loss_factory import LossFactory
from skynet_ml.utils import EarlyStopping
from skynet_ml.utils.batching import DefaultMiniBatchCreator
from skynet_ml.utils.model_utils import create_layer_from_config
from typing import Optional, Union, List
import numpy as np
import csv 

class Sequential: 
    """
    A class representing a sequential neural network model.

    Attributes
    ----------
    layers : list
        List of neural network layers.

    Methods
    -------
    add(layer: Layer):
        Adds a layer to the sequential model.
    compile(loss: Union[str, Loss], regularizer: Optional[Union[str, Regularizer]], optimizer: Union[str, Optimizer], learning_rate: Optional[float]):
        Configures the model for training.
    forward(x: np.array) -> np.array:
        Computes the output of the model for the given input.
    backward(dLda_prev: np.array) -> np.array:
        Computes the gradients for the model using backpropagation.
    fit(xtrain: np.array, ytrain: np.array, xval: Optional[np.array], yval: Optional[np.array], metrics: Optional[List[Union[str, Metric]]], epochs: int, batch_size: int, early_stopping: Optional[EarlyStopping], save_training_history_in: Optional[str]):
        Trains the model on the provided data.
    predict(x: np.array, task_type: str, threshold: float) -> np.array:
        Predicts labels or values for the provided input.
    get_weights() -> List[np.array]:
        Returns the weights of the model's layers.
    set_weights(all_weights):
        Sets the weights for the model's layers.
    get_config():
        Returns the configuration of the model.
    from_config(config):
        Creates a model instance from a given configuration.
    """
    
    def __init__(self):
        """Initializes the Sequential model with an empty list of layers."""
        self.layers = []
        
        
    def add(self, layer: Layer):
        """
        Adds a layer to the sequential model.

        Parameters
        ----------
        layer : Layer
            The layer to be added.

        Raises
        ------
        ValueError
            If it's the first layer and input_dim is not defined.
        """
        # If it's the first layer and input_shape is not defined, raise an error.
        if not self.layers and not hasattr(layer, 'input_dim'):
            raise ValueError('First layer must have input_dim attribute.')
        
        # If it's not the first layer, set the input_dim of the layer to the output_dim of the previous layer.
        if self.layers:
            layer.input_dim = self.layers[-1].n_units
            
        # Add the layer to the list of layers.
        self.layers.append(layer)
        
        
    def compile(self,
                loss: Union[str, Loss],
                regularizer: Optional[Union[str, Regularizer]] = None,
                optimizer: Union[str, Optimizer] = "adam", 
                learning_rate: Optional[float] = None,
                ) -> None:
        """
        Configures the model for training.

        Parameters
        ----------
        loss : Union[str, Loss]
            Loss function or its string representation.
        regularizer : Optional[Union[str, Regularizer]], optional
            Regularizer or its string representation.
        optimizer : Union[str, Optimizer], optional
            Optimizer or its string representation. Defaults to "adam".
        learning_rate : Optional[float], optional
            Learning rate for the optimizer.

        """
        # set the loss, regularizer, and optimizer
        self.loss = LossFactory().get_loss(loss)
        self.regularizer = RegularizerFactory().get_regularizer(regularizer)
        self.optimizer = OptimizerFactory().get_optimizer(optimizer, learning_rate)
        
        # initialize the weights and biases of each layer
        for layer in self.layers:
            layer.initialize()
            
            
    def forward(self, x: np.array) -> np.array:
        """
        Conducts a forward pass through the entire network.

        Parameters:
        - x (np.array): The input array to the network.

        Returns:
        - np.array: The output of the network after the forward pass.

        Raises:
        - ValueError: If no layers are defined in the network.
        """
        # If no layers are defined, raise an error.
        if not self.layers:
            raise ValueError('No layers defined.')
        
        # Forward pass through each layer
        for layer in self.layers:
            x = layer.forward(x)
            
        return x
    
    
    def backward(self, dLda_prev: np.array) -> np.array:
        """
        Conducts a backward pass through the entire network.

        Parameters:
        - dLda_prev (np.array): The gradient of the loss w.r.t. the network's output.

        Returns:
        - np.array: The gradient of the loss w.r.t. the network's input.

        Raises:
        - ValueError: If no layers are defined in the network.
        """
        # If no layers are defined, raise an error.
        if not self.layers:
            raise ValueError('No layers defined.')
        
        # Backward pass through each layer
        for layer in reversed(self.layers):
            dLda_prev = layer.backward(dLda_prev)
            
        return dLda_prev
    
    
    def _save_best_weights(self) -> None:
        """
        Saves the model's best weights if the current loss is better than the best recorded loss.

        Note: It assumes that self.current_loss, self.best_loss, and other necessary attributes exist.
        """
        # If weights have not been updated, return.
        if not self.weights_updated:
            return
        
        # Save the weights to the best weights if the current loss is better than the best loss.
        if self.current_loss < self.best_loss:
            self.best_weights = self.get_weights()
            self.best_loss = self.current_loss
            
        # Set the weights_updated flag to False.
        self.weights_updated = False
        
        
    def _initialize_training(self,
                             xtrain: np.array,
                             ytrain: np.array,
                             metrics: Optional[list] = None,
                             xval: Optional[np.array] = None,
                             yval: Optional[np.array] = None,
                            ) -> None:
        """
        Initializes training attributes for the model.

        Parameters:
        - xtrain, ytrain (np.array): Training data and labels.
        - metrics (list, optional): List of metrics to track during training.
        - xval, yval (np.array, optional): Validation data and labels.
        
        Raises:
        - ValueError: If provided training data or labels are empty.
        """
        # if no x_train or y_train is provided, raise an error
        if not xtrain.size or not ytrain.size:
            raise ValueError("x_train or y_train is empty. Please provide non-empty arrays for training.")
        
        # Initialize best loss and best weights
        self.best_loss = float('inf')
        self.best_weights = None
        
        # Initialize loss and validation loss lists
        self.loss_epochs = []
        self.val_loss_epochs = [] if xval is not None and yval is not None else None
        
        # Initialize metrics and validation metrics dictionaries
        if metrics:
            self.metrics_epochs = {metric: [] for metric in metrics}
            self.val_metrics_epochs = {metric: [] for metric in metrics} if xval is not None and yval is not None else None
        
        
    def _handle_epoch_init(self, metrics: Optional[list] = None) -> None:
        """
        Handles initialization for a new training epoch.

        Parameters:
        - metrics (list, optional): List of metrics to track during training.

        Returns:
        - tuple: Contains loss_batches and metrics_batches initialized for the new epoch.
        """
        self.weights_updated = False
        loss_batches = []
        metrics_batches = {metric: [] for metric in metrics} if metrics else None
        
        return loss_batches, metrics_batches
        
        
    def _process_mini_batch(self,
                            x_mini: np.array,
                            y_mini: np.array,
                            loss_batches: list,
                            metrics_batches: Optional[dict] = None,
                            metrics: Optional[List[Metric]] = None,
                            regularizer: Optional[Regularizer] = None,
                            ) -> None:
        """
        Processes a single mini-batch during training.

        Parameters:
        - x_mini, y_mini (np.array): Mini-batch data and labels.
        - loss_batches (list): List to accumulate batch losses.
        - metrics_batches (dict, optional): Dictionary to accumulate batch metrics.
        - metrics (List[Metric], optional): Metrics to compute.
        - regularizer (Regularizer, optional): Regularizer for weights (e.g. L2 regularization).
        """
        # Forward pass, compute loss, and append to loss_batches
        yhat_mini = self.forward(x_mini)
        loss_mini = self.loss.compute(yhat_mini, y_mini)
        
        # Compute regularization loss and add to loss_mini
        if regularizer:
            loss_mini += np.sum(regularizer.forward(layer.weights) for layer in self.layers)
            
        loss_batches.append(loss_mini)
        
        # Compute metrics and append to metrics_batches
        if metrics:
            for metric in metrics:
                metric_value = metric.compute(yhat_mini, y_mini)
                metrics_batches[metric].append(metric_value)
                
        # Compute the gradient of the loss with respect to the output of the last layer and backpropagate
        dLda = self.loss.gradient(yhat_mini, y_mini)
        self.backward(dLda)
        
        # Compute the gradient of the regularization loss with respect to the output of each layer 
        if regularizer:
            for layer in self.layers:
                layer.dweights += self.regularizer.backward(layer.weights)
        
        # Update the weights
        self.optimizer.step(self.layers)
        self.weights_updated = True
        
        
    def _handle_val_data(self, xval: np.array, yval: np.array, metrics: Optional[List[Metric]] = None):
        """
        Processes validation data to compute loss and metrics.

        Parameters:
        - xval, yval (np.array): Validation data and labels.
        - metrics (List[Metric], optional): Metrics to compute for validation data.

        Returns:
        - float: Validation loss.
        """
        yhat_val = self.forward(xval)
        val_loss = self.loss.compute(yhat_val, yval)
        
        if metrics:
            for metric in metrics:
                metric_value = metric.compute(yhat_val, yval)
                self.val_metrics_epochs[metric].append(metric_value)
                
        return val_loss
    
    
    def _initialize_writer(self, 
                           save_training_history_in: Optional[str] = None,
                           xval: Optional[np.array] = None,
                           yval: Optional[np.array] = None,
                           metrics: Optional[List[Union[str, Metric]]] = None,
                          ) -> None:
        """
        Initializes a CSV writer to save training history.

        Parameters:
        - save_training_history_in (str, optional): Path to save the training history.
        - xval, yval (np.array, optional): Validation data and labels.
        - metrics (List[Union[str, Metric]], optional): Metrics names or objects.

        Returns:
        - file object: File handle to the opened CSV (None if not saving history).
        """
        self.writer = None
        if save_training_history_in is not None:
            f = open(save_training_history_in, 'w', newline='')
            self.writer = csv.writer(f)
            headers = ['epoch', 'loss']
            
            if xval is not None and yval is not None:
                headers.extend(['val_loss'])
            
            if metrics:
                headers.extend(metric.name for metric in metrics)
                
            if xval is not None and yval is not None:
                if metrics:
                    headers.extend(f'val_{metric.name}' for metric in metrics)
                    
            self.writer.writerow(headers)
            
            return f
        
        return None
            
            
    def _print_epoch_info(self, epoch: int, epochs: int) -> None:
        """
        Prints the loss, validation loss, and metrics for a specific epoch.
        
        Parameters:
        - epoch (int): The current epoch.
        - epochs (int): Total number of epochs.
        """
        base_str = f"epoch: {epoch+1}/{epochs} - loss:{self.loss_epochs[-1]:.6f}"
        
        if hasattr(self, 'val_loss_epochs') and self.val_loss_epochs is not None:
            base_str += f" - val_loss:{self.val_loss_epochs[-1]:.6f}"
            
        if hasattr(self, 'metrics_epochs') and self.metrics_epochs is not None:
            for metric in self.metrics_epochs:
                base_str += f" - {metric.name}: {self.metrics_epochs[metric][-1]:.6f}"
                
        if hasattr(self, 'val_metrics_epochs') and self.val_metrics_epochs is not None:
            for metric in self.val_metrics_epochs:
                base_str += f" - val_{metric.name}: {self.val_metrics_epochs[metric][-1]:.6f}"
                
        print(base_str)
            
            
    def _write_to_csv(self, epoch: int) -> None:
        """
        Writes the epoch's loss, validation loss, and metrics data to a CSV file.
        
        Parameters:
        - epoch (int): The current epoch.
        """
        if hasattr(self, "writer") and self.writer is not None:
            row_data = [epoch+1, self.loss_epochs[-1]]
            
            if hasattr(self, "val_loss_epochs"):
                row_data.append(self.val_loss_epochs[-1])
                
            if hasattr(self, "metrics_epochs"):
                row_data.extend([self.metrics_epochs[metric][-1] for metric in self.metrics_epochs])
                
            if hasattr(self, "val_metrics_epochs"):
                row_data.extend([self.val_metrics_epochs[metric][-1] for metric in self.val_metrics_epochs])
                
            self.writer.writerow(row_data)
            
            
    def fit(self,
            xtrain: np.array,
            ytrain: np.array,
            xval: Optional[np.array] = None,
            yval: Optional[np.array] = None,
            metrics: Optional[List[Union[str, Metric]]] = None,
            epochs: int = 1,
            batch_size: int = 8,
            early_stopping: Optional[EarlyStopping] = None,
            save_training_history_in: Optional[str] = None
            ) -> None:
        """
        Trains the neural network on the given data for a specified number of epochs.

        Parameters:
        - xtrain, ytrain (np.array): Training data and labels.
        - xval, yval (np.array, optional): Validation data and labels.
        - metrics (list, optional): Metrics to compute during training.
        - epochs (int, default=1): Number of epochs to train.
        - batch_size (int, default=8): Batch size for mini-batch gradient descent.
        - early_stopping (EarlyStopping, optional): Criteria for early stopping.
        - save_training_history_in (str, optional): Path to save the training history.

        Raises:
        - ValueError: If metrics are not valid or not supported.
        """
        # Initialize metrics objects, training necessary objects, and writer
        metrics = [METRICS_MAP[metric]() if isinstance(metric, str) else metric for metric in metrics] if metrics else None
        self._initialize_training(xtrain, ytrain, metrics, xval, yval)
        f = self._initialize_writer(save_training_history_in, xval, yval, metrics)
        
        try:
            
            for i in range(epochs):
                loss_batches, metrics_batches = self._handle_epoch_init(metrics) # initialize loss and metrics batches
                mini_batches = DefaultMiniBatchCreator.create(xtrain, ytrain, batch_size) # create mini batches
                
                for batch in mini_batches:
                    x_mini, y_mini = batch
                    self._process_mini_batch(x_mini, y_mini, loss_batches, metrics_batches, metrics, self.regularizer) # train on mini batch
                    
                loss_epoch = np.mean(loss_batches) # compute loss for epoch
                self.loss_epochs.append(loss_epoch) 
                self.current_loss = loss_epoch
                
                if xval is not None and yval is not None:
                    val_loss = self._handle_val_data(xval, yval, metrics) # compute validation loss
                    self.val_loss_epochs.append(val_loss)
                    self.current_loss = val_loss
                    
                self._save_best_weights() # save best weights
                
                # compute metrics for epoch
                if metrics:
                    for metric in metrics:
                        metric_epoch_value = np.mean(metrics_batches[metric])
                        self.metrics_epochs[metric].append(metric_epoch_value)
                        
                # print epoch info and write to csv
                self._print_epoch_info(i, epochs)
                self._write_to_csv(i)
                
                # check if early stopping should be triggered
                if early_stopping:
                    if early_stopping.should_stop(self.current_loss):
                        print("\nEarly stopping triggered. Ending training.")
                        break
                    
        finally:
            if hasattr(self, "writer") and self.writer is not None:
                f.close() # close the csv writer if it exists
                
                
    def predict(self, x: np.array, task_type: str = "binary", threshold: float = 0.5) -> np.array:
        """
        Predicts labels for the given input data.

        Parameters:
        - x (np.array): Input data.
        - task_type (str, default="binary"): Type of classification task. Options: 'binary', 'multiclass', 'multilabel'.
        - threshold (float, default=0.5): Threshold for classification.

        Returns:
        - np.array: Predicted labels.

        Raises:
        - ValueError: If `task_type` is not one of 'binary', 'multiclass', or 'multilabel'.
        """
        yhat = self.forward(x)
        
        if task_type in ["binary", "multilabel"]:
            yhat_labels = (yhat > threshold).astype(float)
        elif task_type == "multiclass":
            yhat_labels = np.argmax(yhat, axis=1)
        else:
            raise ValueError(f"task_type must be one of 'binary', 'multiclass', or 'multilabel'. Got {task_type}.")
            
        return yhat_labels
    
    
    def evaluate(self, x: np.array, y: np.array, metrics: Optional[List[Union[str, Metric]]] = None) -> dict:
        """
        Evaluates the model on the given data.

        Parameters:
        - x, y (np.array): Data and labels.
        - metrics (list, optional): Metrics to compute.

        Returns:
        - dict: Dictionary containing the computed metrics.
        """
        metrics = [METRICS_MAP[metric]() if isinstance(metric, str) else metric for metric in metrics] if metrics else None
        loss = self.loss.compute(self.forward(x), y)
        metrics_dict = {"loss": loss}
        
        if metrics:
            for metric in metrics:
                metric_value = metric.compute(self.forward(x), y)
                metrics_dict[metric.name] = metric_value
                
        return metrics_dict
    
    
    def get_weights(self) -> List[np.array]:
        """
        Retrieves the weights of all the layers in the network.

        Returns:
        - List[np.array]: List containing weights of all layers.
        """
        weights = {}
        for idx, layer in enumerate(self.layers):
            weights[f"layer_{idx}"] = layer.get_weights()
        return weights
    
    
    def set_weights(self, all_weights):
        """
        Sets the weights for all the layers in the network.

        Parameters:
        - all_weights (dict): Dictionary containing weights for each layer.
        """
        for idx, layer in enumerate(self.layers):
            layer_key = f"layer_{idx}"
            if layer_key in all_weights:
                layer.set_weights(all_weights[layer_key])
                
                
    def get_config(self):
        """
        Retrieves the configuration of the neural network.

        Returns:
        - dict: Dictionary containing the configuration of the network.
        """
        return {
            "name": "Sequential",
            "layers": [layer.get_config() for layer in self.layers]
        }
        

    @classmethod
    def from_config(cls, config):        
        """
        Creates a NeuralNetwork instance from a given configuration.

        Parameters:
        - config (dict): Configuration dictionary.

        Returns:
        - NeuralNetwork: Instantiated neural network based on the provided configuration.
        """
        model = cls()
        for layer_config in config["layers"]: 
            layer = create_layer_from_config(layer_config)  
            model.add(layer)
            
        return model
