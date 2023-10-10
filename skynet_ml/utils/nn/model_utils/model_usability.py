from skynet_ml.utils.factories import ActivationsFactory, InitializersFactory
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def create_layer_from_config(layer_config):
    """
    Create a layer instance based on its configuration dictionary.

    Args:
    - layer_config (dict): Configuration dictionary of the layer.

    Returns:
    - BaseLayer: An instance of the appropriate layer class based on the configuration.

    Raises:
    - ValueError: If the layer name specified in the configuration is unsupported.
    """
    
    # Handle Dense layer
    if layer_config["name"] == "Dense":
        from skynet_ml.nn.layers.dense import Dense
        
        activation = ActivationsFactory().get_object(layer_config["activation"])
        initializer = InitializersFactory().get_object(layer_config["initializer"]) 
        
        return Dense(
            n_units=layer_config["n_units"],
            activation=activation,
            initializer=initializer,
            has_bias=layer_config["has_bias"],
            input_dim=layer_config["input_dim"]
        )
    
    raise ValueError(f"Unsupported layer name: {layer_config['name']}")


def load_model(filename: str):
    """
    Loads a neural network model from a specified filename.

    Args:
    - filename (str): The path and name of the file from where the model will be loaded.

    Returns:
    - model: The loaded neural network model, reconstructed from the saved data.
    """

    # Import Sequential here to avoid the circular import at module level
    from skynet_ml.nn.models.sequential import Sequential
    
    # Load model data
    model_data = joblib.load(filename)
    
    # Reconstruct model and set weights
    model = Sequential.from_config(model_data["config"])  # Assumes a from_config() classmethod in Sequential
    model.set_weights(model_data["weights"])
    
    return model


def save_model(model, filename: str) -> None:
    """
    Saves the given neural network model to a specified filename using joblib.

    Args:
    - model: The neural network model to save. Expected to be of type Sequential.
    - filename (str): The path and name of the file where the model will be saved.

    Raises:
    - ValueError: If the provided model is not an instance of Sequential.
    """

    # Import Sequential here to avoid the circular import at module level
    from skynet_ml.nn.models.sequential import Sequential
    
    # Check if the passed model is an instance of Sequential
    if not isinstance(model, Sequential):
        raise ValueError("The model must be an instance of Sequential.")
    
    model_data = {
        "config": model.get_config(),  
        "weights": model.get_weights()
    }
    
    # Save model using joblib
    joblib.dump(model_data, filename)
    
    
def plot_model(model, save_in=None):
    """
    Plots the model architecture as a graph.
    """
    i = 1
    output_str = ''

    for layer in model.layers:
        layer_config = layer.get_config()
        
        layer_type = layer_config["name"]
        units = layer_config["n_units"]
        activation = layer_config["activation"]
        initializer = layer_config["initializer"]
        has_bias = layer_config["has_bias"]
        input_dim = layer_config["input_dim"]
        
        item = tuple([layer_type, units, activation, initializer, has_bias, input_dim])
        str1 = "Layer " + str(i) + "\n | Layer Type: {:<10}\n | Units: {:<10}\n | Activation: {:<10}\n | Initializer: {:<10}\n | Has Bias: {:<10}\n | Input Dim: {:<10}\n".format(*item)
        
        output_str += str1
        
        i += 1
        
        if layer == model.layers[-1]:
            break
        
        arrow_str = '\n       |\n       |\n       |\n       V\n\n'
        output_str += arrow_str

    # Decide if you want to print or save to a file based on the save_in parameter
    if save_in:
        with open(save_in, 'w') as f:
            f.write(output_str)
    else:
        print(output_str)
        
        
def plot_training_history(file_name, save_in=None):
    """
    Plots the training and validation loss across epochs from a CSV file.
    """
    
    # Load the CSV
    df = pd.read_csv(file_name)
    
    # Create a new figure
    fig = go.Figure()

    # Add the training loss line
    fig.add_trace(go.Scatter(x=df['epoch'], y=df['loss'], mode='lines',
                             name='Training Loss',
                             line=dict(color='DodgerBlue', width=2)))

    # Add the validation loss line if it exists in the CSV
    if 'val_loss' in df.columns:
        fig.add_trace(go.Scatter(x=df['epoch'], y=df['val_loss'], mode='lines',
                                 name='Validation Loss',
                                 line=dict(color='red', width=2, dash='dash')))
    
    # Background gradient colors
    max_loss = df[['loss', 'val_loss']].max().max()  # Consider the max from both losses
    min_loss = df[['loss', 'val_loss']].min().min()  # Consider the min from both losses
    third_loss = (max_loss - min_loss) / 3
    two_third_loss = 2 * third_loss + min_loss

    fig.update_layout(
        title='Loss Across Epochs',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray')
    )

    # Red background for high loss
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0, x1=1, y0=2/3, y1=1,  # Adjust to span the top third of the graph
        fillcolor="Red",
        opacity=0.2,
        layer="below",
        line_width=0
    )
    
    # Yellow background for medium loss
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0, x1=1, y0=1/3, y1=2/3,  # Adjust to span the middle third of the graph
        fillcolor="DarkOrange",
        opacity=0.2,
        layer="below",
        line_width=0
    )
    
    # Green background for low loss
    fig.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0, x1=1, y0=0, y1=1/3,  # Adjust to span the bottom third of the graph
        fillcolor="LimeGreen",
        opacity=0.2,
        layer="below",
        line_width=0
)
    
    # Save the plot if the save_in parameter is provided
    if save_in:
        fig.write_image(save_in, format='png')
        fig.show()
    else:
        fig.show()
