from skynet_ml.utils.factories import RegularizersFactory, OptimizersFactory, LossesFactory, MetricsFactory
from skynet_ml.utils.nn.batches import DefaultMiniBatchCreator
from skynet_ml.utils.nn.early_stopping import EarlyStopping
from skynet_ml.nn.regularizers.base import BaseRegularizer
from skynet_ml.nn.optimizers.base import BaseOptimizer
from skynet_ml.nn.layers.base import BaseLayer
from skynet_ml.metrics.base import BaseMetric
from skynet_ml.nn.losses.base import BaseLoss
from typing import Optional, Union, List
import pandas as pd
import numpy as np
import json


class Sequential:
    """
    Sequential model class. Simple feedforward neural network.
    """    
    
    
    def __init__(self) -> None:
        """
        Initialize the model.
        """        
        self.layers = []


    def add(self, layer: BaseLayer) -> None:
        """
        Add a layer to the model.

        Args:
            layer (BaseLayer): Layer to add to the model.

        Raises:
            ValueError: If the layer does not have an input_dim attribute and the model does not have any layers.
        """        
        if not self.layers and not hasattr(layer, "input_dim"):
            raise ValueError("Input layer must have input_dim attribute.")
        
        if self.layers:
            layer.input_dim = self.layers[-1].n_units
            
        self.layers.append(layer)


    def compile(self, loss: Union[str, BaseLoss], regularizer: Optional[Union[str, BaseRegularizer]] = None, optimizer: Union[str, BaseOptimizer] = "adam") -> None:
        """
        Compiles the model by initializing the loss, regularizer, and optimizer.

        Args:
            loss (Union[str, BaseLoss]): Loss function to use.
            regularizer (Optional[Union[str, BaseRegularizer]], optional): Regularizer to use. Defaults to None.
            optimizer (Union[str, BaseOptimizer], optional): Optimizer to use. Defaults to "adam".
        """        

        self.loss = LossesFactory().get_object(loss)
        self.optimizer = OptimizersFactory().get_object(optimizer) 
        self.regularizer = RegularizersFactory().get_object(regularizer) if regularizer else None
        
        for layer in self.layers:
            layer.initialize()
            
        self.fix_logits()
            
            
    def fix_logits(self) -> None:        
        """
        Fixes the logits flag in the loss function if the last layer is linear.
        """        
        if "crossentropy" in str(self.loss.name):
            if self.layers[-1].activation.name == "linear":
                self.loss.from_logits = True
            else:
                self.loss.from_logits = False
                
                

    def forward(self, x: np.array) -> np.array:
        """
        Computes the forward pass of the model.

        Args:
            x (np.array): Input data.

        Returns:
            np.array: Output of the model.
        """        
        for layer in self.layers:
            x = layer.forward(x)
            
        return x
    
    
    def backward(self, dl_da_previous: np.array) -> np.array:
        """
        Performs the backward pass of the model.

        Args:
            dl_da_previous (np.array): Gradient of the loss with respect to the output of the last layer.

        Returns:
            np.array: Output of the backward pass.
        """        
        for layer in reversed(self.layers):
            dl_da_previous = layer.backward(dl_da_previous)
            
        return dl_da_previous


    def save_best_weights(self) -> None:
        """
        Saves the best weights of the model.
        """        
        if not self.weights_updated:
            return
        
        if self.current_loss < self.best_loss:
            self.best_weights = self.get_weights()
            self.best_loss = self.current_loss
            
        self.weights_updated = False
        
        
    def get_weights(self) -> List[np.array]:
        """
        Gets the weights of the model.

        Returns:
            List[np.array]: List of weights.
        """        
        weights = {}
        
        for idx, layer in enumerate(self.layers):
            weights[f"layer_{idx}"] = layer.get_weights()
            
        return weights
    

    def set_weights(self, all_weights):
        """
        Sets the weights of the model.

        Args:
            all_weights (dict): Dictionary of weights. 
        """        
        for idx, layer in enumerate(self.layers):
            layer_key = f"layer_{idx}"
            
            if layer_key in all_weights:
                layer.set_weights(all_weights[layer_key])
                

    def initialize_training(self, metrics: Optional[list] = None, x_val: Optional[np.array] = None, y_val: Optional[np.array] = None) -> None:
        """
        Initializes the training of the model.
        """        
        
        # Initialize best loss and weights
        self.best_loss = float('inf')
        self.best_weights = None
        
        # Initialize losses lists
        self.loss_epochs = []
        self.val_loss_epochs = [] if x_val is not None and y_val is not None else None
        
        # Initialize metrics lists
        self.metrics_epochs = {metric: [] for metric in metrics} if metrics else None
        self.val_metrics_epochs = {metric: [] for metric in metrics} if x_val is not None and y_val is not None and metrics else None
        

    def handle_epoch_init(self, metrics: Optional[list] = None) -> Union[list, dict]:
        """
        Initializes the epoch.
        """        
        self.weights_updated = False
        loss_batches = []
        metrics_batches = {metric: [] for metric in metrics} if metrics else None
        
        return loss_batches, metrics_batches


    def process_mini_batch(self, x_mini: np.array, y_mini: np.array, loss_batches: list, metrics_batches: Optional[dict] = None, metrics: Optional[List[BaseMetric]] = None, regularizer: Optional[BaseRegularizer] = None) -> None:
        """
        Processes a mini batch. Computes the forward pass, loss, and metrics. Computes the backward pass and updates the weights.
        """        
        
        # Forward pass, compute loss, and append to loss_batches
        y_hat_mini = self.forward(x_mini)
        loss_mini = self.loss.compute(y_mini, y_hat_mini)
        
        # Compute the regularization loss and add to loss_mini
        if regularizer:
            loss_mini += np.sum(regularizer.forward(layer.weights) for layer in self.layers)
            
        # Append loss_mini to loss_batches
        loss_batches.append(loss_mini)
        
        # Compute metrics and append to metrics_batches
        if metrics: 
            for metric in metrics:
                metric_value = metric.compute(y_mini, y_hat_mini)
                metrics_batches[metric].append(metric_value)
                
        # Compute the gradient of the loss with respect to the output of the last layer and backpropagate
        dl_da = self.loss.gradient(y_mini, y_hat_mini)
        self.backward(dl_da)
        
        # Compute the gradient of the regularization loss with respect to the output of each layer 
        if regularizer:
            for layer in self.layers:
                layer.d_weights += self.regularizer.backward(layer.weights)
        
        # Update the weights
        self.optimizer.step(self.layers)
        self.weights_updated = True
        
        
    def handle_val_data(self, x_val: np.array, y_val: np.array, metrics: Optional[List[BaseMetric]] = None) -> float:
        """
        Uses the model on the validation data and computes the validation loss and metrics.
        """        
        y_hat_val = self.forward(x_val)
        val_loss = self.loss.compute(y_val, y_hat_val)
        
        if metrics:
            for metric in metrics:
                metric_value = metric.compute(y_val, y_hat_val)
                self.val_metrics_epochs[metric].append(metric_value)
                
        return val_loss


    def print_epoch_info(self, epoch: int, epochs: int) -> None:
        """
        Prints the epoch info.
        """        
        base_str = f"epoch: {epoch+1}/{epochs} - loss:{self.loss_epochs[-1]:.6f}"
        
        if hasattr(self, 'val_loss_epochs') and self.val_loss_epochs is not None:
            base_str += f" - val_loss:{self.val_loss_epochs[-1]:.6f}"
            
        if hasattr(self, 'metrics_epochs') and self.metrics_epochs is not None:
            for metric in self.metrics_epochs:
                base_str += f" - {metric.name.split('_')[0]}: {self.metrics_epochs[metric][-1]:.6f}"
                
        if hasattr(self, 'val_metrics_epochs') and self.val_metrics_epochs is not None:
            for metric in self.val_metrics_epochs:
                base_str += f" - val_{metric.name.split('_')[0]}: {self.val_metrics_epochs[metric][-1]:.6f}"
                
        print(base_str)
        
    
    def get_epoch_info_as_dict(self, epoch: int) -> dict:
        """
        Saves the epoch info as a dictionary.
        """        
        info_dict = {}
        info_dict["epoch"] = epoch + 1
        info_dict["loss"] = self.loss_epochs[-1]

        if hasattr(self, 'val_loss_epochs') and self.val_loss_epochs is not None:
            info_dict["val_loss"] = self.val_loss_epochs[-1]

        if hasattr(self, 'metrics_epochs') and self.metrics_epochs is not None:
            for metric in self.metrics_epochs:
                info_dict[metric.name.split('_')[0]] = self.metrics_epochs[metric][-1]

        if hasattr(self, 'val_metrics_epochs') and self.val_metrics_epochs is not None:
            for metric in self.val_metrics_epochs:
                info_dict[f"val_{metric.name.split('_')[0]}"] = self.val_metrics_epochs[metric][-1]

        return info_dict
        
        
    def fix_task_type(self, metrics: Optional[List[BaseMetric]]) -> None:
        """
        Fixes the task type of the metrics.
        """        
        if self.layers[-1].n_units > 1 and "categorical" in str(self.loss.name):
            task_type = "multiclass"
        
        elif self.layers[-1].n_units > 1 and "binary" in str(self.loss.name):
            task_type = "multilabel"
            
        else:
            task_type = "binary"
            
        for metric in metrics:
            metric.task_type = task_type
            
            
    def fit(
            self, 
            x_train: np.array, 
            y_train: np.array, 
            x_val: Optional[np.array] = None, 
            y_val: Optional[np.array] = None, 
            metrics: Optional[List[Union[str, BaseMetric]]] = None, 
            epochs: int = 1, batch_size: int = 1, 
            early_stopping: Optional[EarlyStopping] = None, 
            save_training_history_in: Optional[str] = None,
            save_activations_in: Optional[str] = None,
            save_gradients_in: Optional[str] = None
        ) -> None:
        """
        Fits the model.

        Args:
            x_train (np.array): training data.
            y_train (np.array): training labels.
            x_val (Optional[np.array], optional): validation data. Defaults to None.
            y_val (Optional[np.array], optional): validation labels. Defaults to None.
            metrics (Optional[List[Union[str, BaseMetric]]], optional): Metrics to compute and plot. Defaults to None.
            epochs (int, optional): Number of epochs to train. Defaults to 1.
            batch_size (int, optional): Number of samples per batch. Defaults to 1.
            early_stopping (Optional[EarlyStopping], optional): EarlyStopping object. Defaults to None.
            save_training_history_in (Optional[str], optional): Path to where training history will be saved. Defaults to None.
            save_activations_in (Optional[str], optional): Path to where activation values will be saved. Defaults to None.
            save_gradients_in (Optional[str], optional): Path to where gradient values will be saved. Defaults to None.
        """        
        
        # Initialize training
        metrics = [MetricsFactory().get_object(metric) for metric in metrics] if metrics else None
        self.fix_task_type(metrics) if metrics else None
        self.initialize_training(metrics, x_val, y_val)
        all_data = []
        dict_of_activations = {}
        dict_of_gradients = {}
            
        for i in range(epochs):
            
            # Handle epoch initialization, create batches and new lists 
            loss_batches, metrics_batches = self.handle_epoch_init(metrics)
            mini_batches = DefaultMiniBatchCreator.create(x_train, y_train, batch_size) 
            
            # Process each mini batch
            for batch in mini_batches:
                x_mini, y_mini = batch
                self.process_mini_batch(x_mini, y_mini, loss_batches, metrics_batches, metrics, self.regularizer)
                
            # Compute and save epoch data, save best weights, print epoch info
            loss_epoch = np.mean(loss_batches)
            self.loss_epochs.append(loss_epoch)
            self.current_loss = loss_epoch
            
            # Use model on validation data if provided and save val loss
            if x_val is not None and y_val is not None:
                val_loss = self.handle_val_data(x_val, y_val, metrics)
                self.val_loss_epochs.append(val_loss)
                self.current_loss = val_loss
                
            # Save best weights
            self.save_best_weights()
            
            # Compute and save metrics 
            if metrics:
                for metric in metrics:
                    metric_epoch_value = np.mean(metrics_batches[metric])
                    self.metrics_epochs[metric].append(metric_epoch_value)
                    
            # Print epoch info and save epoch data
            self.print_epoch_info(i, epochs)
            all_data.append(self.get_epoch_info_as_dict(i))
            
            
            # Save training history to csv file
            if save_training_history_in:
                df = pd.DataFrame(all_data)
                df.to_csv(save_training_history_in, index=False)
                
            
            # Save activations 
            if save_activations_in:
                j = 1
                dict_activations = {}
                dict_activations["epoch"] = i
                for layer in self.layers:
                    dict_activations[f"layer_{j}"] = layer.a.flatten().tolist()
                    j += 1
                dict_of_activations[i] = dict_activations
            
            # Save gradients
            if save_gradients_in:
                j = 1
                dict_gradients = {}
                dict_gradients["epoch"] = i
                for layer in self.layers:
                    dict_gradients[f"layer_{j}_weights"] = layer.d_weights.flatten().tolist()
                    dict_gradients[f"layer_{j}_bias"] = layer.d_bias.flatten().tolist()
                    j += 1
                dict_of_gradients[i] = dict_gradients
            
            # check if early stopping should be triggered
            if early_stopping:
                if early_stopping.should_stop(self.current_loss):
                    print("\nEarly stopping triggered. Ending training.")
                    break
                
        # Save activations to json file
        if save_activations_in:
            with open(save_activations_in, "w") as f:
                json.dump(dict_of_activations, f)
            
        # Save gradients to json file
        if save_gradients_in:
            with open(save_gradients_in, "w") as f:
                json.dump(dict_of_gradients, f)
        
        
    def predict(self, x: np.array, one_hotted_output: bool = False, threshold: float = 0.5) -> np.array:
        """
        Predicts the output of the model.
        """        
        y_pred = self.forward(x)
        
        if one_hotted_output:
            if "binary_crossentropy" in str(self.loss.name):
                y_pred = (y_pred > threshold).astype(int)
            elif "categorical_crossentropy" in str(self.loss.name):
                y_pred = np.where(np.greater_equal(y_pred,  np.max(y_pred, axis=1)[:, None]), 1, 0)
            
        return y_pred
    
    
    def evaluate(self, x: np.array, y: np.array, metrics: Optional[List[Union[str, BaseMetric]]] = None) -> dict:
        """
        Evaluates the model on the given data with the given metrics.
        """        
        y_hat = self.forward(x)
        loss = self.loss.compute(y, y_hat)
        metrics_dict = {"loss": loss}
        
        
        if metrics:
            metrics = [MetricsFactory().get_object(metric) for metric in metrics] if metrics else None
            self.fix_task_type(metrics) if metrics else None
            for metric in metrics:
                metric_value = metric.compute(y, y_hat)
                metrics_dict[metric.name] = metric_value
                
        return metrics_dict
    