from skynet_ml.utils.factories import RegularizersFactory, OptimizersFactory, LossesFactory, MetricsFactory
from skynet_ml.utils.nn.model_utils import create_layer_from_config
from skynet_ml.utils.nn.batches import DefaultMiniBatchCreator
from skynet_ml.nn.regularizers.regularizer import Regularizer
from skynet_ml.utils.nn.early_stopping import EarlyStopping
from skynet_ml.nn.optimizers.optimizer import Optimizer
from skynet_ml.metrics.metric import Metric
from skynet_ml.nn.layers.layer import Layer
from skynet_ml.nn.losses.loss import Loss
from typing import Optional, Union, List
import numpy as np
import csv 


class Sequential:
    """
    The Sequential model is a linear stack of layers to create a neural network.

    The Sequential class is designed to facilitate the easy creation, training, and evaluation of simple neural networks 
    with a sequential layer structure. It provides a range of methods for assembling networks, configuring their learning
    process, training them on data, evaluating their performance, and generating predictions on new data.

    Attributes
    ----------
    layers : List[Layer]
        A list holding the sequence of Layer instances comprising the network.
    loss : Loss
        The loss function used during training to compare the network's predictions to the true values.
    regularizer : Regularizer, optional
        The regularization term added to the loss to prevent overfitting (default is None).
    optimizer : Optimizer
        The optimization algorithm used to minimize the loss function during training.
    best_loss : float
        The lowest value of the loss function achieved on the training data during training.
    best_weights : List[np.array]
        The values of the network’s weights that correspond to its best_loss.
    loss_epochs : List[float]
        List containing the value of the loss function at each epoch of training.
    val_loss_epochs : List[float], optional
        List containing the value of the loss function at each epoch of validation, if validation data is provided (default is None).
    metrics_epochs : Dict[Metric, List[float]], optional
        Dictionary mapping Metric objects to lists containing their computed values at each epoch of training, if metrics are provided (default is None).
    val_metrics_epochs : Dict[Metric, List[float]], optional
        Dictionary mapping Metric objects to lists containing their computed values at each epoch of validation, if validation data and metrics are provided (default is None).

    Methods
    -------
    add(layer: Layer) -> None:
        Add a layer instance to the sequential stack.
    compile(loss: Union[str, Loss], regularizer: Optional[Union[str, Regularizer]] = None, optimizer: Union[str, Optimizer] = "adam") -> None:
        Configures the model for training.
    fit(x_train: np.array, y_train: np.array, x_val: Optional[np.array] = None, y_val: Optional[np.array] = None, metrics: Optional[List[Union[str, Metric]]] = None, epochs: int = 1, batch_size: int = 1, early_stopping: Optional[EarlyStopping] = None, save_training_history_in: Optional[str] = None) -> None:
        Trains the model for a fixed number of epochs on the training data, optionally evaluating it on validation data.
    evaluate(x: np.array, y: np.array, metrics: Optional[List[Union[str, Metric]]] = None) -> dict:
        Evaluates the model's performance on the provided data.
    predict(x: np.array, one_hotted_output: bool = False, threshold: float = 0.5) -> np.array:
        Generates output predictions for the input samples.
    (Additional methods are not listed here for brevity but should be documented individually.)

    Notes
    -----
    - The Sequential model is appropriate for networks comprising a linear stack of layers where each layer has weights that depend only on the previous layer.
    - The layers are added using the `add` method in the order they should process the inputs.
    - The `compile` method must be called before training to configure the learning process.
    - For training, use the `fit` method, which also supports evaluation on validation data and early stopping.

    Examples
    --------
    Here is a simple example of using the Sequential model:

    >>> from some_module import Dense, Softmax
    >>> model = Sequential()
    >>> model.add(Dense(64, activation='relu', input_dim=100))
    >>> model.add(Dense(10, activation='softmax'))
    >>> model.compile(loss='categorical_crossentropy', optimizer='sgd')
    >>> model.fit(x_train, y_train, epochs=10)
    """
    
    
    
    def __init__(self,) -> None:
        """
        Initialize a new Sequential object.

        The Sequential class is used to define a linear stack of network layers which can be used
        to create a neural network.

        Attributes
        ----------
        layers : list
            An empty list that will store Layer objects added to the Sequential model.
        """
        self.layers = []



    def add(self, layer: Layer) -> None:
        """
        Add a Layer object to the Sequential model.

        Parameters
        ----------
        layer : Layer
            The layer to be added to the model.

        Raises
        ------
        ValueError
            If it's the first layer being added and it doesn't have the 'input_dim' attribute defined.

        Notes
        -----
        - If it's the first layer being added, the layer must have the 'input_dim' attribute set.
        - For subsequent layers, their 'input_dim' is automatically set to the 'n_units' of the previous layer.
        """
        if not self.layers and not hasattr(layer, "input_dim"):
            raise ValueError("Input layer must have input_dim attribute.")
        if self.layers:
            layer.input_dim = self.layers[-1].n_units
        self.layers.append(layer)



    def compile(self, loss: Union[str, Loss], regularizer: Optional[Union[str, Regularizer]] = None, optimizer: Union[str, Optimizer] = "adam") -> None:
        """
        Configure the Sequential model for training.

        Parameters
        ----------
        loss : Union[str, Loss]
            The loss function to be used during training. It could either be a string (name of the loss function)
            or an instance of a Loss class.
        regularizer : Union[str, Regularizer], optional
            The regularization method to be used. It could either be a string (name of the regularizer)
            or an instance of a Regularizer class. Default is None.
        optimizer : Union[str, Optimizer], optional
            The optimization algorithm to be used. It could either be a string (name of the optimizer)
            or an instance of an Optimizer class. Default is 'adam'.

        Notes
        -----
        - This method initializes the layers added to the model and sets up the loss, regularizer, and optimizer
          for the training process.
        - The layers are initialized by calling their 'initialize' method.
        """
        self.loss = self._process_loss(loss)
        self.regularizer = RegularizersFactory().get_object(regularizer) 
        self.optimizer = OptimizersFactory().get_object(optimizer)
    
        for layer in self.layers:
            layer.initialize()



    def _process_loss(self, loss: Union[str, Loss]) -> Loss:
        """
        Process the input loss parameter and return a Loss object.

        This method helps in converting a loss given as a string to its corresponding
        Loss object using a factory. It also sets the from_logits parameter if applicable.

        Parameters
        ----------
        loss : Union[str, Loss]
            Loss parameter that can either be a string (name of the loss function)
            or an instance of a Loss class.

        Returns
        -------
        Loss
            The Loss object corresponding to the input loss parameter.

        Notes
        -----
        The method uses the _should_use_logits method to determine the value of from_logits
        if the loss is given as a string.
        """
        if isinstance(loss, str):
            from_logits = self._should_use_logits(loss)
            return LossesFactory().get_object(loss, from_logits=from_logits)
        return LossesFactory().get_object(loss)



    def _should_use_logits(self, loss: str) -> bool:
        """
        Determine if logits should be used based on the loss function and activation.

        Parameters
        ----------
        loss : str
            The name of the loss function.

        Returns
        -------
        bool
            Returns False if the loss is cross entropy and the activation of the last layer is
            either Softmax or Sigmoid. Otherwise, returns True.

        Notes
        -----
        This method is used internally to set the from_logits parameter while processing loss functions.
        """
        cross_entropies = ["categorical_crossentropy", "binary_crossentropy"]
        if loss in cross_entropies and self.layers[-1].get_config()["activation"] in ["Softmax", "Sigmoid"]:
            return False
        return True



    def forward(self, x: np.array) -> np.array:
        """
        Compute the forward pass through all layers of the model.

        Parameters
        ----------
        x : np.array
            Input data, a Numpy array of shape (batch_size, input_dim).

        Returns
        -------
        np.array
            Output data computed by the forward pass through the model. The shape of the output
            depends on the configuration of the model's layers.

        Notes
        -----
        This method sequentially calls the forward method of each layer in self.layers, passing
        the output of one layer as the input to the next.
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    
    
    def backward(self, dl_da_previous: np.array) -> np.array:
        """
        Perform the backward pass through all layers of the model.

        Parameters
        ----------
        dl_da_previous : np.array
            Gradient of the loss with respect to the output.

        Returns
        -------
        np.array
            Gradient of the loss with respect to the input.

        Notes
        -----
        This method sequentially calls the backward method of each layer in reversed order,
        passing the gradient from one layer to the next.
        """
        for layer in reversed(self.layers):
            dl_da_previous = layer.backward(dl_da_previous)
        return dl_da_previous



    def save_best_weights(self) -> None:
        """
        Save the model's best weights during training.

        This method should be called during training after each epoch. It saves the model's weights
        if they correspond to the lowest loss encountered so far.

        Notes
        -----
        - If the weights haven't been updated, the method returns without doing anything.
        - It updates the best_weights attribute with the current weights if the current loss
          is lower than the best loss encountered so far.
        - After saving, it resets the weights_updated flag to False.
        """
        if not self.weights_updated:
            return
        if self.current_loss < self.best_loss:
            self.best_weights = self.get_weights()
            self.best_loss = self.current_loss
        self.weights_updated = False



    def initialize_training(self,
                            x_train: np.array,
                            y_train: np.array,
                            metrics: Optional[list] = None,
                            x_val: Optional[np.array] = None,
                            y_val: Optional[np.array] = None) -> None:
        """
        Initialize training-related attributes of the model.

        Parameters
        ----------
        x_train : np.array
            Training input data.
        y_train : np.array
            Training target data.
        metrics : list, optional
            List of metric names to be evaluated during training. Default is None.
        x_val : np.array, optional
            Validation input data. Default is None.
        y_val : np.array, optional
            Validation target data. Default is None.

        Raises
        ------
        ValueError
            If no training data is provided.

        Notes
        -----
        This method initializes or resets various attributes related to the training process,
        including best loss, best weights, loss history, and metrics history.
        """
        if not x_train.size or not y_train.size:
            raise ValueError("Training data must be provided.")
        
        self.best_loss = float('inf')
        self.best_weights = None
        
        self.loss_epochs = []
        self.val_loss_epochs = [] if x_val is not None and y_val is not None else None
        
        if metrics:
            self.metrics_epochs = {metric: [] for metric in metrics}
            self.val_metrics_epochs = {metric: [] for metric in metrics} if x_val is not None and y_val is not None else None
        
            
            
    def handle_epoch_init(self, metrics: Optional[list] = None) -> Union[list, dict]:
        """
        Initialize attributes for a new training epoch.

        Parameters
        ----------
        metrics : list, optional
            List of metric names to be evaluated during training. Default is None.

        Returns
        -------
        tuple
            A tuple containing a list for batch-wise loss values and a dictionary for batch-wise metric values.

        Notes
        -----
        This method initializes or resets the weights_updated flag, the loss_batches list, and the metrics_batches dictionary.
        """
        self.weights_updated = False
        loss_batches = []
        metrics_batches = {metric: [] for metric in metrics} if metrics else None
        
        return loss_batches, metrics_batches
    
    
    
    def process_mini_batch(self,
                           x_mini: np.array,
                           y_mini: np.array, 
                           loss_batches: list,
                           metrics_batches: Optional[dict] = None,
                           metrics: Optional[List[Metric]] = None,
                           regularizer: Optional[Regularizer] = None) -> None:
        """
        Process a mini-batch of data during training.

        Parameters
        ----------
        x_mini : np.array
            Mini-batch of input data.
        y_mini : np.array
            Mini-batch of target data.
        loss_batches : list
            List to store loss values computed for each mini-batch.
        metrics_batches : dict, optional
            Dictionary to store metric values computed for each mini-batch. Default is None.
        metrics : List[Metric], optional
            List of Metric objects to be computed for each mini-batch. Default is None.
        regularizer : Regularizer, optional
            Regularizer object to be applied to the weights during training. Default is None.

        Notes
        -----
        This method performs the forward and backward passes, computes the loss and metrics,
        and updates the model's weights for a given mini-batch of data.
        """
        
        # Forward pass, compute loss, and append to loss_batches
        y_hat_mini = self.forward(x_mini)
        loss_mini = self.loss.compute(y_mini, y_hat_mini)
        
        
        if regularizer:
            loss_mini += np.sum(regularizer.forward(layer.weights) for layer in self.layers)
            
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
        
        
        
    def handle_val_data(self, x_val: np.array, y_val: np.array, metrics: Optional[List[Metric]] = None) -> float:
        """
        Process the validation data after each training epoch.

        Parameters
        ----------
        x_val : np.array
            Validation input data.
        y_val : np.array
            Validation target data.
        metrics : List[Metric], optional
            List of Metric objects to be computed for the validation data. Default is None.

        Returns
        -------
        float
            Validation loss computed for the given validation data.

        Notes
        -----
        This method performs the forward pass on the validation data, computes the validation loss
        and validation metrics, and appends the metric values to val_metrics_epochs.
        """
        
        y_hat_val = self.forward(x_val)
        val_loss = self.loss.compute(y_val, y_hat_val)
        
        if metrics:
            for metric in metrics:
                metric_value = metric.compute(y_val, y_hat_val)
                self.val_metrics_epochs[metric].append(metric_value)
                
        return val_loss
    
    
    
    def initialize_writer(self,
                          save_training_history_in: Optional[str] = None,
                          x_val: Optional[np.array] = None,
                          y_val: Optional[np.array] = None,
                          metrics: Optional[List[Union[str, Metric]]] = None) -> None:
        """
        Initialize the CSV writer for logging training history.

        Parameters
        ----------
        save_training_history_in : str, optional
            File path where the training history CSV file should be saved. Default is None.
        x_val : np.array, optional
            Validation input data. Default is None.
        y_val : np.array, optional
            Validation target data. Default is None.
        metrics : List[Union[str, Metric]], optional
            List of metrics to be logged. Default is None.

        Notes
        -----
        Initializes a CSV writer and writes the headers of the CSV file for training history.
        Headers include epoch, loss, validation loss (if validation data is provided), and any specified metrics.
        """
        
        self.writer = None
        if save_training_history_in is not None:
            f = open(save_training_history_in, 'w', newline='')
            self.writer = csv.writer(f)
            headers = ['epoch', 'loss']
            
            if x_val is not None and y_val is not None:
                headers.extend(['val_loss'])
                
            if metrics:
                headers.extend(metric.name for metric in metrics)
                
            if x_val is not None and y_val is not None:
                if metrics:
                    headers.extend(f'val_{metric.name}' for metric in metrics)
                    
            self.writer.writerow(headers)
            
            return f
        
        return None
    
    
    
    def print_epoch_info(self, epoch: int, epochs: int) -> None:
        """
        Print information about the training process for the current epoch.

        Parameters
        ----------
        epoch : int
            The current epoch number (0-indexed).
        epochs : int
            Total number of epochs for training.

        Notes
        -----
        The method generates and prints a string containing information about the current epoch,
        including the epoch number, loss, validation loss, and values of any tracked metrics.
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
        
        
        
    def write_to_csv(self, epoch: int) -> None:
        """
        Write training history for the current epoch to a CSV file.

        Parameters
        ----------
        epoch : int
            The current epoch number (0-indexed).

        Notes
        -----
        The method writes a row to the CSV file containing information about the current epoch,
        including the epoch number, loss, validation loss, and values of any tracked metrics.
        The CSV writer should have been previously initialized by calling `initialize_writer`.
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
            
            
            
    def _process_metrics(self, metrics: Optional[List[Union[str, Metric]]] = None) -> None:
        """
        Process the provided metrics into a list of corresponding Metric objects.

        Parameters
        ----------
        metrics : List[Union[str, Metric]], optional
            List of metrics to be processed, specified either as strings or Metric objects. Default is None.

        Returns
        -------
        List[Metric]
            List of Metric objects corresponding to the provided metrics. If no metrics are provided,
            an empty list is returned.

        Notes
        -----
        This method converts the list of metrics into a list of corresponding Metric objects. For string metrics,
        it determines the appropriate task type (binary, multiclass, or multilabel) based on the last layer's activation
        and the specified loss function, and retrieves the corresponding Metric object. For Metric objects, it directly
        retrieves the corresponding Metric object.
        
        This method uses two internal functions: `determine_task_type` and `get_metric_object` to facilitate this processing.
        `determine_task_type` calculates the task type based on the metric name and the configurations of the last layer
        and the loss. `get_metric_object` utilizes the `determine_task_type` function if the metric is provided as a string,
        and retrieves the corresponding Metric object, else it directly retrieves the Metric object if it is already provided.
        """
        if not metrics:
            return []  # Return an empty list if no metrics are provided
            
        def determine_task_type(metric: str) -> str:
            """
            This function determines the task type based on the metric name and the configurations
            of the last layer and loss.
            """
            activation = self.layers[-1].get_config()["activation"]  # Get activation of last layer
            n_units = self.layers[-1].n_units  # Get number of units of last layer
            loss_name = self.loss.name  # Get the name of the loss
            
            # Check conditions to determine the task type
            is_softmax = activation == "softmax"
            is_sigmoid_and_single_unit = activation == "sigmoid" and n_units == 1

            # Map metric to task type based on conditions
            if metric in ["accuracy", "precision", "recall", "fscore"]:
                if is_softmax or loss_name == "categorical_crossentropy":
                    return "multiclass"
                if is_sigmoid_and_single_unit or (loss_name == "binary_crossentropy" and n_units == 1):
                    return "binary"
                return "multilabel"
            return None  # Return None if task type cannot be determined


        def get_metric_object(metric: Union[str, Metric]) -> Metric:
            """
            This function obtains a metric object. If the input metric is a string, it determines
            the task type (if applicable) and gets the corresponding metric object. Otherwise,
            it gets the metric object directly.
            """
            if isinstance(metric, str):  # If metric is a string
                task_type = determine_task_type(metric)  # Determine its task type
                # Get metric object with or without task type based on its determination
                return MetricsFactory().get_object(metric, task_type=task_type) if task_type else MetricsFactory().get_object(metric)
            return MetricsFactory().get_object(metric)  # Get metric object directly if it's not a string
            
        # Return a list of metric objects processed from the input metrics
        return [get_metric_object(metric) for metric in metrics]
                
                
            
    def fit(self,
            x_train: np.array,
            y_train: np.array,
            x_val: Optional[np.array] = None,
            y_val: Optional[np.array] = None,
            metrics: Optional[List[Union[str, Metric]]] = None,
            epochs: int = 1,
            batch_size: int = 1,
            early_stopping: Optional[EarlyStopping] = None,
            save_training_history_in: Optional[str] = None,
            ) -> None:
        """
        Fit the model to the training data, optionally using validation data and early stopping.

        Parameters
        ----------
        x_train : np.array
            Numpy array of training data inputs.
        y_train : np.array
            Numpy array of training data targets.
        x_val : np.array, optional
            Numpy array of validation data inputs. Default is None.
        y_val : np.array, optional
            Numpy array of validation data targets. Default is None.
        metrics : List[Union[str, Metric]], optional
            List of metrics to evaluate the model. Metrics can be strings or Metric objects. Default is None.
        epochs : int, optional
            Number of epochs to train the model. Default is 1.
        batch_size : int, optional
            Number of samples per gradient update. Default is 1.
        early_stopping : EarlyStopping, optional
            EarlyStopping object to end training when a monitored quantity has stopped improving. Default is None.
        save_training_history_in : str, optional
            File path where training history CSV file should be saved. Default is None.

        Notes
        -----
        This method trains the model for a fixed number of epochs, iterating on the data in mini-batches.
        The method processes the training data, handles epoch initialization, processes each mini-batch,
        computes and logs loss and specified metrics, optionally processes validation data, saves best weights,
        prints epoch information, writes training history to CSV file, and implements early stopping if specified.
        """
        
        metrics = self._process_metrics(metrics)
        self.initialize_training(x_train, y_train, metrics, x_val, y_val)
        f = self.initialize_writer(save_training_history_in, x_val, y_val, metrics)
        
        try:
            for i in range(epochs):
                loss_batches, metrics_batches = self.handle_epoch_init(metrics)
                mini_batches = DefaultMiniBatchCreator.create(x_train, y_train, batch_size) 
                
                for batch in mini_batches:
                    x_mini, y_mini = batch
                    self.process_mini_batch(x_mini, y_mini, loss_batches, metrics_batches, metrics, self.regularizer)
                    
                loss_epoch = np.mean(loss_batches)
                self.loss_epochs.append(loss_epoch)
                self.current_loss = loss_epoch
                
                if x_val is not None and y_val is not None:
                    val_loss = self.handle_val_data(x_val, y_val, metrics)
                    self.val_loss_epochs.append(val_loss)
                    self.current_loss = val_loss
                    
                self.save_best_weights()
                
                if metrics:
                    for metric in metrics:
                        metric_epoch_value = np.mean(metrics_batches[metric])
                        self.metrics_epochs[metric].append(metric_epoch_value)
                        
                self.print_epoch_info(i, epochs)
                self.write_to_csv(i)
                
                if early_stopping:
                    if early_stopping.should_stop(self.current_loss):
                        print("\nEarly stopping triggered. Ending training.")
                        break
                
        finally:
            if hasattr(self, "writer") and self.writer is not None:
                f.close()
                
                
                
    def predict(self, x: np.array, one_hotted_output: bool = False, threshold: float = 0.5) -> np.array:
        """
        Generate output predictions for the input samples.

        Parameters
        ----------
        x : np.array
            Numpy array of input samples to generate predictions.
        one_hotted_output : bool, optional
            Whether to return predictions as one-hot encoded arrays. Useful for classification tasks.
            Default is False.
        threshold : float, optional
            Threshold for predicting class labels in binary classification tasks when `one_hotted_output` is True.
            Ignored for non-binary classification tasks. Default is 0.5.

        Returns
        -------
        np.array
            Array of predictions generated by the model for the input samples.

        Notes
        -----
        This method performs a forward pass of the input data through the model to generate predictions.
        For binary classification tasks with `one_hotted_output` set to True, it applies a threshold to
        generate class labels. For multi-class classification tasks with `one_hotted_output` set to True,
        it converts the output probabilities to one-hot encoded format.

        Examples
        --------
        >>> model.predict(x_test)
        array([[0.1], [0.9], [0.4], ...])
        
        >>> model.predict(x_test, one_hotted_output=True)
        array([[0], [1], [0], ...])

        >>> model.predict(x_test, one_hotted_output=True, threshold=0.4)
        array([[1], [1], [1], ...])
        """
        
        y_pred = self.forward(x)
        if one_hotted_output:
            if self.loss.name == "binary_crossentropy":
                y_pred = (y_pred > threshold).astype(int)
            elif self.loss.name == "categorical_crossentropy":
                y_pred = np.where(np.greater_equal(y_pred,  np.max(y_pred, axis=1)[:, None]), 1, 0)
            
        return y_pred
    
    
    
    def evaluate(self, x: np.array, y: np.array, metrics: Optional[List[Union[str, Metric]]] = None) -> dict:
        """
        Evaluate the model's performance on the provided data.

        Parameters
        ----------
        x : np.array
            Numpy array of input data.
        y : np.array
            Numpy array of target data.
        metrics : List[Union[str, Metric]], optional
            List of metrics to evaluate the model. Metrics can be strings or Metric objects. Default is None.

        Returns
        -------
        dict
            Dictionary containing the computed loss and additional metrics (if provided) on the input data.

        Notes
        -----
        This method performs a forward pass of the input data through the model, computes the loss,
        processes the provided metrics, computes each metric value, and returns a dictionary containing
        the loss and the computed metrics values.

        Examples
        --------
        >>> model.evaluate(x_test, y_test, metrics=['accuracy', 'precision'])
        {'loss': 0.23, 'accuracy': 0.95, 'precision': 0.96}
        """
        
        y_hat = self.forward(x)
        loss = self.loss.compute(y, y_hat)
        metrics_dict = {"loss": loss}
        
        if metrics:
            metrics = self._process_metrics(metrics)
            for metric in metrics:
                metric_value = metric.compute(y, y_hat)
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
            "layers": [layer.get_config() for layer in self.layers],
            "loss": self.loss,
            "regularizer": self.regularizer,
            "optimizer": self.optimizer
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
        model.loss = config["loss"]
        model.regularizer = config["regularizer"]
        model.optimizer = config["optimizer"]
            
        return model
